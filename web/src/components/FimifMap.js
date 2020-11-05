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
import {ForceEdgeBundling} from "../js/d3-ForceEdgeBundling"



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

        // const colorDomain = [...Array(props.labelNum).keys()]
        // const colorRange = ["#bab0ab","#f28e2c","#4e79a7","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f", "#e15759","#5E4FA2"];
        // const colorScale =  d3.scaleOrdinal().domain(colorDomain).range(colorRange);
        
        
        threshold.current = 0.5;   // will be changed (with slidebar interaction)
        let powList;
        let thresholdVal, maxVal, domain;
        if (rMode === "fg") {
            powList = embeddedData.map(d => {
                return Math.pow(Math.pow(d.false_value[0], 2) + Math.pow(d.false_value[1], 2), 0.5)
            });
            powList.sort()
            console.log(powList)
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
        console.log(domain)

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
                            console.log(d.missing_points);
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
            let clusters = dbscan.run(missing_distortion_points_simplified, 3, 1);
            let cluster_indices = []
            for (let cluster_idx in clusters) {
                let current_cluster_indices = [];
                let cluster = clusters[cluster_idx]
                for (let i in cluster) {
                    current_cluster_indices.push(missing_distortion_points_index[cluster[i]]);
                }
                cluster_indices.push(current_cluster_indices);
            }
            console.log(cluster_indices);

            cluster_indices.forEach(cluster => {
                console.log(cluster)
                let cluster_coor = cluster.map(i => {
                    return embeddedData[i].coor;
                })
                console.log(cluster_coor)
                let contour = hull(cluster_coor, 20);
                contour.push(contour[0]);
                console.log(contour);
                cluster_info.push({
                    indices: cluster,
                    contour: contour
                })
            })
            console.log(cluster_info)
            

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
                                // .curve(d3.curveMonotoneX)
                                .curve(d3.curveCatmullRom)
                                (datum.contour)
               )

            let connections = []
            missing_distortion_points.forEach(d => {
                Object.keys(d.missing_points).forEach(e => {
                    if (d.idx >= parseInt(e))
                        connections.push([d.idx, parseInt(e)])
                })
            })
            console.log(connections);

            let node_data = {}
            let edge_data = connections.map(connection => {
                if (!(connection[0] in node_data)) node_data[connection[0]] = embeddedData[connection[0]].coor;
                if (!(connection[1] in node_data)) node_data[connection[1]] = embeddedData[connection[1]].coor;
                return {"source": connection[0], "target": connection[1]}
            });
            node_data = Object.keys(node_data).map(key => {
                return {
                    "x": node_data[key][0],
                    "y": node_data[key][1]
                }
            })

            console.log(edge_data)
            console.log(node_data)


            let fbundling = ForceEdgeBundling
            console.log(fbundling)
            
            
            svg.selectAll(".connection")
               .data(connections)
               .enter()
               .append("path")
               .attr("class", "connection")
               .attr("fill", "none")
               .attr("stroke-width", 1)
               .attr("opacity", 0.1)
               .attr("stroke", "red")
               .attr("d",  datum => d3.line()
                    .x(d => xScale(embeddedData[d].coor[0]))
                    .y(d => yScale(embeddedData[d].coor[1]))
                    // .curve(d3.curveMonotoneX)
                    .curve(d3.curveBundle.beta(1))
                    (datum)
               
               )

            // const distFunc = Clustering.distFunc.euclidean;
            // const cluster = new Clustering(missing_distortion_points_simplified, distFunc);
            // console.log(cluster.getTree());
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

