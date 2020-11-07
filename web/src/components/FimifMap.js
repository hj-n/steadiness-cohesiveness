import React from 'react';
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import { orange } from '@material-ui/core/colors';
import hull from 'hull.js'
import clustering from 'density-clustering'
import {Helmet} from "react-helmet";


// Function for force edge bundling
let ForceEdgeBundling = function () {
    var data_nodes = {}, // {'nodeid':{'x':,'y':},..}
        data_edges = [], // [{'source':'nodeid1', 'target':'nodeid2'},..]
        compatibility_list_for_edge = [],
        subdivision_points_for_edge = [],
        K = 0.1, // global bundling constant controlling edge stiffness
        S_initial = 0.1, // init. distance to move points
        P_initial = 1, // init. subdivision number
        P_rate = 2, // subdivision rate increase
        C = 6, // number of cycles to perform
        I_initial = 90, // init. number of iterations for cycle
        I_rate = 0.6666667, // rate at which iteration number decreases i.e. 2/3
        compatibility_threshold = 0.6,
        eps = 1e-6;

    let P, spring_force, electrostatic_force
    /*** Geometry Helper Methods ***/
    function vector_dot_product(p, q) {
        return p.x * q.x + p.y * q.y;
    }

    function edge_as_vector(P) {
        return {
            'x': data_nodes[P.target].x - data_nodes[P.source].x,
            'y': data_nodes[P.target].y - data_nodes[P.source].y
        }
    }

    function edge_length(e) {
        // handling nodes that are on the same location, so that K/edge_length != Inf
        if (Math.abs(data_nodes[e.source].x - data_nodes[e.target].x) < eps &&
            Math.abs(data_nodes[e.source].y - data_nodes[e.target].y) < eps) {
            return eps;
        }

        return Math.sqrt(Math.pow(data_nodes[e.source].x - data_nodes[e.target].x, 2) +
            Math.pow(data_nodes[e.source].y - data_nodes[e.target].y, 2));
    }

    function custom_edge_length(e) {
        return Math.sqrt(Math.pow(e.source.x - e.target.x, 2) + Math.pow(e.source.y - e.target.y, 2));
    }

    function edge_midpoint(e) {
        var middle_x = (data_nodes[e.source].x + data_nodes[e.target].x) / 2.0;
        var middle_y = (data_nodes[e.source].y + data_nodes[e.target].y) / 2.0;

        return {
            'x': middle_x,
            'y': middle_y
        };
    }

    function compute_divided_edge_length(e_idx) {
        var length = 0;

        for (var i = 1; i < subdivision_points_for_edge[e_idx].length; i++) {
            var segment_length = euclidean_distance(subdivision_points_for_edge[e_idx][i], subdivision_points_for_edge[e_idx][i - 1]);
            length += segment_length;
        }

        return length;
    }

    function euclidean_distance(p, q) {
        return Math.sqrt(Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2));
    }

    function project_point_on_line(p, Q) {
        var L = Math.sqrt((Q.target.x - Q.source.x) * (Q.target.x - Q.source.x) + (Q.target.y - Q.source.y) * (Q.target.y - Q.source.y));
        var r = ((Q.source.y - p.y) * (Q.source.y - Q.target.y) - (Q.source.x - p.x) * (Q.target.x - Q.source.x)) / (L * L);

        return {
            'x': (Q.source.x + r * (Q.target.x - Q.source.x)),
            'y': (Q.source.y + r * (Q.target.y - Q.source.y))
        };
    }

    /*** ********************** ***/

    /*** Initialization Methods ***/
    function initialize_edge_subdivisions() {
        for (var i = 0; i < data_edges.length; i++) {
            if (P_initial === 1) {
                subdivision_points_for_edge[i] = []; //0 subdivisions
            } else {
                subdivision_points_for_edge[i] = [];
                subdivision_points_for_edge[i].push(data_nodes[data_edges[i].source]);
                subdivision_points_for_edge[i].push(data_nodes[data_edges[i].target]);
            }
        }
    }

    function initialize_compatibility_lists() {
        for (var i = 0; i < data_edges.length; i++) {
            compatibility_list_for_edge[i] = []; //0 compatible edges.
        }
    }

    function filter_self_loops(edgelist) {
        var filtered_edge_list = [];

        for (var e = 0; e < edgelist.length; e++) {

            if (data_nodes[edgelist[e].source].x != data_nodes[edgelist[e].target].x ||
                data_nodes[edgelist[e].source].y != data_nodes[edgelist[e].target].y) { //or smaller than eps
                filtered_edge_list.push(edgelist[e]);
            }
        }

        return filtered_edge_list;
    }

    /*** ********************** ***/

    /*** Force Calculation Methods ***/
    function apply_spring_force(e_idx, i, kP) {
        var prev = subdivision_points_for_edge[e_idx][i - 1];
        var next = subdivision_points_for_edge[e_idx][i + 1];
        var crnt = subdivision_points_for_edge[e_idx][i];
        var x = prev.x - crnt.x + next.x - crnt.x;
        var y = prev.y - crnt.y + next.y - crnt.y;

        x *= kP;
        y *= kP;

        return {
            'x': x,
            'y': y
        };
    }

    function apply_electrostatic_force(e_idx, i) {
        var sum_of_forces = {
            'x': 0,
            'y': 0
        };
        var compatible_edges_list = compatibility_list_for_edge[e_idx];

        for (var oe = 0; oe < compatible_edges_list.length; oe++) {
            var force = {
                'x': subdivision_points_for_edge[compatible_edges_list[oe]][i].x - subdivision_points_for_edge[e_idx][i].x,
                'y': subdivision_points_for_edge[compatible_edges_list[oe]][i].y - subdivision_points_for_edge[e_idx][i].y
            };

            if ((Math.abs(force.x) > eps) || (Math.abs(force.y) > eps)) {
                var diff = (1 / Math.pow(custom_edge_length({
                    'source': subdivision_points_for_edge[compatible_edges_list[oe]][i],
                    'target': subdivision_points_for_edge[e_idx][i]
                }), 1));

                sum_of_forces.x += force.x * diff;
                sum_of_forces.y += force.y * diff;
            }
        }

        return sum_of_forces;
    }


    function apply_resulting_forces_on_subdivision_points(e_idx, P, S) {
        var kP = K / (edge_length(data_edges[e_idx]) * (P + 1)); // kP=K/|P|(number of segments), where |P| is the initial length of edge P.
        // (length * (num of sub division pts - 1))
        var resulting_forces_for_subdivision_points = [{
            'x': 0,
            'y': 0
        }];

        for (var i = 1; i < P + 1; i++) { // exclude initial end points of the edge 0 and P+1
            var resulting_force = {
                'x': 0,
                'y': 0
            };

            spring_force = apply_spring_force(e_idx, i, kP);
            electrostatic_force = apply_electrostatic_force(e_idx, i, S);

            resulting_force.x = S * (spring_force.x + electrostatic_force.x);
            resulting_force.y = S * (spring_force.y + electrostatic_force.y);

            resulting_forces_for_subdivision_points.push(resulting_force);
        }

        resulting_forces_for_subdivision_points.push({
            'x': 0,
            'y': 0
        });

        return resulting_forces_for_subdivision_points;
    }

    /*** ********************** ***/

    /*** Edge Division Calculation Methods ***/
    function update_edge_divisions(P) {
        for (var e_idx = 0; e_idx < data_edges.length; e_idx++) {
            if (P === 1) {
                subdivision_points_for_edge[e_idx].push(data_nodes[data_edges[e_idx].source]); // source
                subdivision_points_for_edge[e_idx].push(edge_midpoint(data_edges[e_idx])); // mid point
                subdivision_points_for_edge[e_idx].push(data_nodes[data_edges[e_idx].target]); // target
            } else {
                var divided_edge_length = compute_divided_edge_length(e_idx);
                var segment_length = divided_edge_length / (P + 1);
                var current_segment_length = segment_length;
                var new_subdivision_points = [];
                new_subdivision_points.push(data_nodes[data_edges[e_idx].source]); //source

                for (var i = 1; i < subdivision_points_for_edge[e_idx].length; i++) {
                    var old_segment_length = euclidean_distance(subdivision_points_for_edge[e_idx][i], subdivision_points_for_edge[e_idx][i - 1]);

                    while (old_segment_length > current_segment_length) {
                        var percent_position = current_segment_length / old_segment_length;
                        var new_subdivision_point_x = subdivision_points_for_edge[e_idx][i - 1].x;
                        var new_subdivision_point_y = subdivision_points_for_edge[e_idx][i - 1].y;

                        new_subdivision_point_x += percent_position * (subdivision_points_for_edge[e_idx][i].x - subdivision_points_for_edge[e_idx][i - 1].x);
                        new_subdivision_point_y += percent_position * (subdivision_points_for_edge[e_idx][i].y - subdivision_points_for_edge[e_idx][i - 1].y);
                        new_subdivision_points.push({
                            'x': new_subdivision_point_x,
                            'y': new_subdivision_point_y
                        });

                        old_segment_length -= current_segment_length;
                        current_segment_length = segment_length;
                    }
                    current_segment_length -= old_segment_length;
                }
                new_subdivision_points.push(data_nodes[data_edges[e_idx].target]); //target
                subdivision_points_for_edge[e_idx] = new_subdivision_points;
            }
        }
    }

    /*** ********************** ***/

    /*** Edge compatibility measures ***/
    function angle_compatibility(P, Q) {
        return Math.abs(vector_dot_product(edge_as_vector(P), edge_as_vector(Q)) / (edge_length(P) * edge_length(Q)));
    }

    function scale_compatibility(P, Q) {
        var lavg = (edge_length(P) + edge_length(Q)) / 2.0;
        return 2.0 / (lavg / Math.min(edge_length(P), edge_length(Q)) + Math.max(edge_length(P), edge_length(Q)) / lavg);
    }

    function position_compatibility(P, Q) {
        var lavg = (edge_length(P) + edge_length(Q)) / 2.0;
        var midP = {
            'x': (data_nodes[P.source].x + data_nodes[P.target].x) / 2.0,
            'y': (data_nodes[P.source].y + data_nodes[P.target].y) / 2.0
        };
        var midQ = {
            'x': (data_nodes[Q.source].x + data_nodes[Q.target].x) / 2.0,
            'y': (data_nodes[Q.source].y + data_nodes[Q.target].y) / 2.0
        };

        return lavg / (lavg + euclidean_distance(midP, midQ));
    }

    function edge_visibility(P, Q) {
        var I0 = project_point_on_line(data_nodes[Q.source], {
            'source': data_nodes[P.source],
            'target': data_nodes[P.target]
        });
        var I1 = project_point_on_line(data_nodes[Q.target], {
            'source': data_nodes[P.source],
            'target': data_nodes[P.target]
        }); //send actual edge points positions
        var midI = {
            'x': (I0.x + I1.x) / 2.0,
            'y': (I0.y + I1.y) / 2.0
        };
        var midP = {
            'x': (data_nodes[P.source].x + data_nodes[P.target].x) / 2.0,
            'y': (data_nodes[P.source].y + data_nodes[P.target].y) / 2.0
        };

        return Math.max(0, 1 - 2 * euclidean_distance(midP, midI) / euclidean_distance(I0, I1));
    }

    function visibility_compatibility(P, Q) {
        return Math.min(edge_visibility(P, Q), edge_visibility(Q, P));
    }

    function compatibility_score(P, Q) {
        return (angle_compatibility(P, Q) * scale_compatibility(P, Q) * position_compatibility(P, Q) * visibility_compatibility(P, Q));
    }

    function are_compatible(P, Q) {
        return (compatibility_score(P, Q) >= compatibility_threshold);
    }

    function compute_compatibility_lists() {
        for (var e = 0; e < data_edges.length - 1; e++) {
            for (var oe = e + 1; oe < data_edges.length; oe++) { // don't want any duplicates
                if (are_compatible(data_edges[e], data_edges[oe])) {
                    compatibility_list_for_edge[e].push(oe);
                    compatibility_list_for_edge[oe].push(e);
                }
            }
        }
    }

    /*** ************************ ***/

    /*** Main Bundling Loop Methods ***/
    var forcebundle = function () {
        var S = S_initial;
        var I = I_initial;
        var P = P_initial;

        initialize_edge_subdivisions();
        initialize_compatibility_lists();
        update_edge_divisions(P);
        compute_compatibility_lists();

        for (var cycle = 0; cycle < C; cycle++) {
            for (var iteration = 0; iteration < I; iteration++) {
                var forces = [];
                for (var edge = 0; edge < data_edges.length; edge++) {
                    forces[edge] = apply_resulting_forces_on_subdivision_points(edge, P, S);
                }
                for (var e = 0; e < data_edges.length; e++) {
                    for (var i = 0; i < P + 1; i++) {
                        subdivision_points_for_edge[e][i].x += forces[e][i].x;
                        subdivision_points_for_edge[e][i].y += forces[e][i].y;
                    }
                }
            }
            // prepare for next cycle
            S = S / 2;
            P = P * P_rate;
            I = I_rate * I;

            update_edge_divisions(P);
            //console.log('C' + cycle);
            //console.log('P' + P);
            //console.log('S' + S);
        }
        return subdivision_points_for_edge;
    };
    /*** ************************ ***/


    /*** Getters/Setters Methods ***/
    forcebundle.nodes = function (nl) {
        if (arguments.length === 0) {
            return data_nodes;
        } else {
            data_nodes = nl;
        }

        return forcebundle;
    };

    forcebundle.edges = function (ll) {
        if (arguments.length === 0) {
            return data_edges;
        } else {
            data_edges = filter_self_loops(ll); //remove edges to from to the same point
        }

        return forcebundle;
    };

    forcebundle.bundling_stiffness = function (k) {
        if (arguments.length === 0) {
            return K;
        } else {
            K = k;
        }

        return forcebundle;
    };

    forcebundle.step_size = function (step) {
        if (arguments.length === 0) {
            return S_initial;
        } else {
            S_initial = step;
        }

        return forcebundle;
    };

    forcebundle.cycles = function (c) {
        if (arguments.length === 0) {
            return C;
        } else {
            C = c;
        }

        return forcebundle;
    };

    forcebundle.iterations = function (i) {
        if (arguments.length === 0) {
            return I_initial;
        } else {
            I_initial = i;
        }

        return forcebundle;
    };

    forcebundle.iterations_rate = function (i) {
        if (arguments.length === 0) {
            return I_rate;
        } else {
            I_rate = i;
        }

        return forcebundle;
    };

    forcebundle.subdivision_points_seed = function (p) {
        if (arguments.length == 0) {
            return P;
        } else {
            P = p;
        }

        return forcebundle;
    };

    forcebundle.subdivision_rate = function (r) {
        if (arguments.length === 0) {
            return P_rate;
        } else {
            P_rate = r;
        }

        return forcebundle;
    };

    forcebundle.compatibility_threshold = function (t) {
        if (arguments.length === 0) {
            return compatibility_threshold;
        } else {
            compatibility_threshold = t;
        }

        return forcebundle;
    };

    /*** ************************ ***/

    return forcebundle;
}

function FimifMap(props) {

    let jsonFileName = props.dataset + "_" + props.method;
    let data = require("../json/" + jsonFileName + ".json");
    let missing_data = require("../json/" + jsonFileName + "_missing.json");
    let false_data = require("../json/" + jsonFileName + "_false.json");
    const embeddedData = data.map((d, i) => {
        let embeddedDatum = {
            "coor": d.emb,                      // 2D Coordinate
            "idx": i,                           // Current Index
            "missing_points": missing_data[i],  // Missing points dictionary
            "false_value": false_data[i]        // Power / direction of false value
        }
        return embeddedDatum;
    });

    console.log(embeddedData)
    // props.method = props.method.replace(".","")


    const [rMode, setRMode] = useState("mg") // true: use max 2D dist point, false: use max ND dist point
    const handleChange = (event) => {
        setRMode(event.target.value);
    };

    const xS = useRef();   // xScale
    const yS = useRef();   // yScale
    const cS = useRef();   // colorScale
    
    const threshold = useRef();    // threshold for determining the saturation of false score


    function getColor(rMode, d, colorScale, thresholdVal, maxVal) {
        let power;
        if(rMode === "fg") {
            power = Math.pow(Math.pow(d.false_value[0], 2) + Math.pow(d.false_value[1], 2), 0.5);
            if (power < thresholdVal) power = thresholdVal
        }
        else {
            let sum = 0;
            power = 0;
            Object.keys(d.missing_points).forEach(key => {
                sum += d.missing_points[key];
            });
            if (sum > 0)
                sum /= Object.keys(d.missing_points).length;
            power = sum * 2 + maxVal;
        }
        return colorScale(power)
    }

    let svg;

    const width = props.width;
    const height = props.height;
    const margin = { hor: props.width / 20, ver: props.height / 20 };

    useEffect(() => {

            
        svg = d3.select("#scatterplot" + props.dataset + props.method)
                    .attr("width", width + margin.hor * 2)
                    .attr("height", height + margin.ver * 2)
                    .append("g")
                    .attr("id", "scatterplot_g" + props.dataset + props.method)
                    .attr("transform", "translate(" + margin.hor + ", " + margin.ver + ")");

    }, [])

    useEffect(() => {


        

        const [minX, maxX] = d3.extent(embeddedData, d => d.coor[0]);
        const [minY, maxY] = d3.extent(embeddedData, d => d.coor[1]);

        const xScale = d3.scaleLinear()
                         .domain([minX, maxX])
                         .range([0, width]);
        
        const yScale = d3.scaleLinear()
                         .domain([minY, maxY])
                         .range([0, height]);

        
        threshold.current = 0.5;   // will be changed (with slidebar interaction)
        let powList;
        let thresholdVal, maxVal, domain;
        if (rMode === "fg") {
            powList = embeddedData.map(d => {
                return Math.pow(Math.pow(d.false_value[0], 2) + Math.pow(d.false_value[1], 2), 0.5)
            });
            powList.sort()
            let thresholdIdx = Math.round(powList.length * threshold.current)
            let maxVal = powList[powList.length - 1]
            thresholdVal = powList[thresholdIdx]
            domain = [maxVal - 1.5 * (maxVal - thresholdVal), maxVal]
        }
        else {
            powList = embeddedData.map(d => {
                let sum = 0;
                Object.keys(d.missing_points).forEach(key => {
                    sum += d.missing_points[key];
                });
                if (sum > 0)
                    return sum / Object.keys(d.missing_points).length;
                else return 0;
            });
            maxVal = Math.max(...powList)
            domain = [0, (maxVal * 3)];
        }
        
        // let domain = [domain[1] - 2*(domain[1] - threshold.current), domain[1]]

        let colorScale = d3.scaleSequential().domain(domain).interpolator(d3.interpolateGreys);

        cS.current = colorScale;
        xS.current = xScale;
        yS.current = yScale;


        
        const radius = 3.5;
        

        svg = d3.select("#scatterplot_g" + props.dataset + props.method);
        
        // ANCHOR Scatterplot rending & interaction
        svg.selectAll("circle")
           .data(embeddedData)
           .join(
               enter => {
                   enter.append("circle")
                        .attr("class", d => "circle" + d.idx.toString())
                        .attr("fill", d => getColor(rMode, d, colorScale, thresholdVal, maxVal) )
                        .attr("cx", d => xScale(d.coor[0]))
                        .attr("cy", d => yScale(d.coor[1]))
                        .style("opacity", 0.8)
                        .attr("r", radius)
                        .on("mouseover", function(event, d) {
                            Object.keys(d.missing_points).forEach(key => {
                                svg.select(".circle" + key)
                                   .attr("fill", "red")
                            })
                            d3.select(this).style("opacity", 1).attr("r", radius * 2);
                        })
                        .on("mouseout", function(event, d) {
                            d3.select(this).style("opacity", 0.8).attr("r", radius); 
                            Object.keys(d.missing_points).forEach(key => {
                                svg.select(".circle" + key)
                                   .attr("fill", d => getColor(rMode, d, colorScale, thresholdVal, maxVal) )
                            })
                            
                        })
               },
               update => {
                   update.attr("fill", d => getColor(rMode, d, colorScale, thresholdVal, maxVal))
               }
           );

        let cluster_info = []

        if(rMode === "fg") {

        }
        else {
            let missing_distortion_points = embeddedData.filter(d => {
                if (Object.keys(d.missing_points).length > 0) return true;
                else return false;
            });
            let missing_distortion_points_index = []
            let missing_distortion_points_simplified = missing_distortion_points.map(d => {
                missing_distortion_points_index.push(d.idx);
                return d.coor;
            });

            let dbscan = new clustering.DBSCAN();
            let clusters = dbscan.run(missing_distortion_points_simplified, 1, 7);
            let cluster_indices = []
            for (let cluster_idx in clusters) {
                let current_cluster_indices = [];
                let cluster = clusters[cluster_idx]
                for (let i in cluster) {
                    current_cluster_indices.push(missing_distortion_points_index[cluster[i]]);
                }
                cluster_indices.push(current_cluster_indices);
            }

            cluster_indices.forEach(cluster => {
                let cluster_coor = cluster.map(i => {
                    return embeddedData[i].coor;
                })
                let contour = hull(cluster_coor, 20);
                contour.push(contour[0]);
                cluster_info.push({
                    indices: cluster,
                    contour: contour
                })
            })
            

            svg.selectAll(".contour")
               .data(cluster_info)
               .enter()
               .append("path")
               .attr("class", "contour")
               .attr("fill", "none")
               .attr("stroke-width", 3)
               .attr("stroke", "black")
               .attr("d", datum => d3.line()
                                .x(d => xScale(d[0]))
                                .y(d => yScale(d[1]))
                                .curve(d3.curveCatmullRom)
                                (datum.contour)
               )

            let connections = []
            let idxToCluster = {}
            cluster_info.forEach((d, i) => {
                let cluster = d.indices;
                cluster.forEach(idx => { idxToCluster[idx] = i;})
            })



            // forming the connections between cluster subset
            // cluster n and cluster m: dictionary key is n_m
            let clusterConnections = {}
            cluster_info.forEach((d, clusterIdx) => {
                let cluster = d.indices;
                cluster.forEach(i => {
                    let missingPointsDict = embeddedData[i].missing_points
                    Object.keys(missingPointsDict).forEach(j => {
                        let otherClusterIdx = idxToCluster[j];
                        if (otherClusterIdx !== undefined) {
                            let key = clusterIdx.toString() + "_" + otherClusterIdx.toString();
                            if (!(key in clusterConnections)) {
                                let newClusterConnection = {}
                                newClusterConnection[clusterIdx.toString()] = [i];
                                newClusterConnection[otherClusterIdx.toString()] = [parseInt(j)];
                                newClusterConnection["score"] = missingPointsDict[j];
                                newClusterConnection["num"] = 1;
                                clusterConnections[key] = newClusterConnection;
                            }
                            else {
                                clusterConnections[key][clusterIdx.toString()].push(i);
                                clusterConnections[key][otherClusterIdx.toString()].push(parseInt(j)); 
                                clusterConnections[key]["score"] += missingPointsDict[j];
                                clusterConnections[key]["num"] += 1;
                            }
                        }
                    })
                })
            })

            // Remove repeatition
            Object.keys(clusterConnections).forEach(key => {
                clusterConnections[key]["score"] /= clusterConnections[key]["num"]
                let cluster = clusterConnections[key];
                Object.keys(cluster).forEach(k => { 
                    if (k !== "score" && k !=="num")
                        cluster[k] = Array.from(new Set(cluster[k]));
                })
            })

            console.log(cluster_info, clusterConnections);


            // data for connection weight visualization
            let connectionsWeight = [];
            let minWeight = Number.MAX_VALUE;
            let maxWeight = Number.MIN_VALUE;
            // make connections
            Object.keys(clusterConnections).forEach(key => {
                
                let cluster = clusterConnections[key];
                let clusterArr = Object.keys(cluster).map(k => cluster[k]);
                let longerArr, shorterArr;

                let weight = clusterConnections[key].score;
                minWeight = weight < minWeight ? weight : minWeight;
                maxWeight = weight > maxWeight ? weight : maxWeight;
                if(Object.keys(clusterArr).length === 4) { // 예외처리
                    let longerIdx = clusterArr[1].length > clusterArr[0].length ? 1 : 0;
                    longerArr = clusterArr[longerIdx];
                    shorterArr = clusterArr[(longerIdx + 1) % 2];
                    longerArr.forEach((d, i) => {
                        let j = i % shorterArr.length;
                        connections.push([d, shorterArr[j]]); 
                        connectionsWeight.push(weight);
                    })
                    
                }
            })

            console.log(connections);
            console.log(connectionsWeight);
            console.log(minWeight, maxWeight);

            let pathOpacityScale = d3.scaleLinear().domain([minWeight, maxWeight]).range([0, 1]);


            let node_data = {}
            let edge_data = connections.map(connection => {
                if (!(connection[0] in node_data)) node_data[connection[0]] = embeddedData[connection[0]].coor;
                if (!(connection[1] in node_data)) node_data[connection[1]] = embeddedData[connection[1]].coor;
                return {"source": connection[0], "target": connection[1]}
            });
            let nodeIdxToDataIdx = {}
            node_data = Object.keys(node_data).map((key, i) => {
                nodeIdxToDataIdx[key] = i;
                return {
                    "x": node_data[key][0],
                    "y": node_data[key][1]
                }
            })
            edge_data = edge_data.map(datum => {
                return {
                    "source": nodeIdxToDataIdx[datum.source],
                    "target": nodeIdxToDataIdx[datum.target]
                }
            })


            let fbundling = ForceEdgeBundling().step_size(0.1)
                                               .compatibility_threshold(0.85)
                                               .nodes(node_data).edges(edge_data)
            let results = fbundling();
            console.log(results)


            svg.selectAll(".connection")
               .data(results)
               .enter()
               .append("path")
               .attr("class", "connection")
               .attr("fill", "none")
               .attr("stroke-width", 1)
               .attr("opacity", (d, i) => pathOpacityScale(connectionsWeight[i]) * 0.05 )
            //    .attr("opacity", 0.02)
               .attr("stroke", "red")
               .attr("d",  datum => d3.line()
                    .x(d => xScale(d.x))
                    .y(d => yScale(d.y))
                    .curve(d3.curveMonotoneX)
                    (datum)
               )      
        }
    }, [rMode])


    return (
        <div style={{width: props.width * 1.1, margin: 5}}>
            <H6>{props.dataset} dataset embedded by {props.method}</H6>
            <RadioWrapper>
                <RadioGroup row aria-label="distortionType" name="DistortionType" value={rMode} onChange={handleChange}>
                    <FormControlLabel value="fg" control={<Radio />} label="False Groups Distortion" />
                    <FormControlLabel value="mg" control={<Radio />} label="Missing Groups Distortion" />
                </RadioGroup>
            </RadioWrapper>
            <svg id={"scatterplot" + props.dataset + props.method}></svg>
        </div>
    )
}

const H6 = styled.h5`
    margin: 3px;
    font-size: 1.2em; 
    text-align: center;
`;

const RadioWrapper = styled.div`
    justify-content: center;
    display: flex;
    alignIterms: 'center';

`

export default FimifMap;

