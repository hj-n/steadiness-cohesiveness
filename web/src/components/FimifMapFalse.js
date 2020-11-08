// Temporary react component For False Case


import React from 'react';
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import hull from 'hull.js'
import clustering from 'density-clustering'


function FimifMap(props) {

    let jsonFileName = props.dataset + "_" + props.method;
    let data = require("../json/" + jsonFileName + ".json");
    let missing_data = require("../json/" + jsonFileName + "_missing.json");
    let false_data = require("../json/" + jsonFileName + "_false.json");
    let knn_data = require("../json/" + jsonFileName + "_knn.json");

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


    const [rMode, setRMode] = useState("fg") // true: use max 2D dist point, false: use max ND dist point
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
            // if (sum > 0)
            //     sum /= Object.keys(d.missing_points).length;
            power = sum * 2 + maxVal;
        }
        return colorScale(power)
    }


    let svg;
    let svgForCircle;
    let svgForPath;
    let svgForEdge;

    const width = props.width;
    const height = props.height;
    const margin = { hor: props.width / 20, ver: props.height / 20 };


    // OnMount
    useEffect(() => {

            
        svg = d3.select("#scatterplot" + props.dataset + props.method)
                    .attr("width", width + margin.hor * 2)
                    .attr("height", height + margin.ver * 2)
                    .append("g")
                    .attr("id", "scatterplot_g" + props.dataset + props.method)
                    .attr("transform", "translate(" + margin.hor + ", " + margin.ver + ")");

        svgForCircle = svg.append("g")
                          .attr("id", "circle_g" + props.dataset + props.method);
        svgForEdge = svg.append("g")
                        .attr("id", "edge_g" + props.dataset + props.method);
        svgForPath = svg.append("g")
                        .attr("id", "path_g" + props.dataset + props.method);

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
                    return sum // / Object.keys(d.missing_points).length;
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


        
        const radius = 3;
        

        svg = d3.select("#scatterplot_g" + props.dataset + props.method);
        svgForCircle = d3.select("#circle_g" + props.dataset + props.method);
        svgForPath = d3.select("#path_g" + props.dataset + props.method);
        svgForEdge = d3.select("#edge_g" + props.dataset + props.method);

        // ANCHOR Scatterplot rending & interaction
        svgForCircle.selectAll("circle")
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
                    .on("mouseenter", function(event, d) {
                        d3.select(this).style("opacity", 1).attr("r", radius * 2);
                        Object.keys(d.missing_points).forEach(key => {
                            svgForCircle.select(".circle" + key)
                                .attr("fill", "red")
                        })
                    })
                    .on("mouseleave", function(event, d) {
                        d3.select(this).style("opacity", 0.8).attr("r", radius); 
                        Object.keys(d.missing_points).forEach(key => {
                            svgForCircle.select(".circle" + key)
                                .attr("fill", d => getColor(rMode, d, colorScale, thresholdVal, maxVal) )
                        })
                        
                    })
            },
            update => {
                update.attr("fill", d => getColor(rMode, d, colorScale, thresholdVal, maxVal))
            }
        );


        let knnArr = knn_data.reduce(function(acc, d, i) {
            d.forEach(e => { if (i < e) acc.push([i, e]); });
            return acc;
        }, [])


        /* ************ */
        // Fg case
        // similarity * power between edges: cosine similairty * averge of two false value
        let knnArrVal = knnArr.map(edge => {
            let leftFalse = embeddedData[edge[0]].false_value;
            let rightFalse = embeddedData[edge[1]].false_value;
            let leftFalseLength = Math.sqrt(Math.pow(leftFalse[0], 2) + Math.pow(rightFalse[1], 2));
            let rightFalseLength = Math.sqrt(Math.pow(rightFalse[0], 2) + Math.pow(rightFalse[1], 2));
            let dotProduct = leftFalse[0] * rightFalse[0] + leftFalse[1] * rightFalse[1];
            let cosineSim = dotProduct / (leftFalseLength * rightFalseLength);
            cosineSim = cosineSim < 0 ? 0 : cosineSim;
            // console.log(leftFalseLength, rightFalseLength, dotProduct, cosineSim);
            let average = (leftFalseLength + rightFalseLength) / 2;
            return cosineSim * average;
        })

        knnArr = knnArr.filter((d, i) => knnArrVal[i] > 0);
        knnArrVal = knnArrVal.filter((d, i) => d > 0);

        console.log(knnArr, knnArrVal);


        let knnDomain = [Math.min(...knnArrVal), Math.max(...knnArrVal)];
        let knnScale = d3.scaleLinear().domain(knnDomain).range([0, 1]);

        svgForEdge.selectAll("path")
                  .data(knnArr)
                  .enter()
                  .append("path")
                  .attr("fill", "none")
                  .attr("stroke-width", 1)
                  .attr("opacity", (d, i) => knnScale(knnArrVal[i]))
                  .attr("stroke", "black")
                  .attr("d",  datum => d3.line()
                    .x(d => xScale(embeddedData[d].coor[0]))
                    .y(d => yScale(embeddedData[d].coor[1]))
                        (datum)
                  );

        svgForPath.selectAll("path")
                  .data(embeddedData)
                  .enter()
                  .append("path")
                  .attr("fill", "none")
                  .attr("stroke-width", 1)
                  .attr("stroke", "red")
                  .attr("opacity", 0.4)
                  .attr("d", d => {
                    let lr = 0.0002;
                    let start = d.coor;
                    let end = [d.coor[0] + d.false_value[0] * lr, d.coor[1] + d.false_value[1] * lr];
                    let data = [start, end];
                    console.log(data);
                    return d3.line()
                             .x(dd => xScale(dd[0]))
                             .y(dd => yScale(dd[1]))
                             (data);
                  });
                  





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
            let clusters = dbscan.run(missing_distortion_points_simplified, 0.9, 4);
            let cluster_indices = []
            for (let cluster_idx in clusters) {
                let current_cluster_indices = [];
                let cluster = clusters[cluster_idx]
                for (let i in cluster) {
                    current_cluster_indices.push(missing_distortion_points_index[cluster[i]]);
                }
                cluster_indices.push(current_cluster_indices);
            }

            // console.log(clusters, cluster_indices)
            // console.log(missing_distortion_points_index)
            let missingIndexSet = new Set(missing_distortion_points_index);
            cluster_indices.forEach(cluster => {
                cluster.forEach(idx => {
                    missingIndexSet.delete(idx);
                })
            })
            console.log(missingIndexSet);
            missingIndexSet.forEach(idx => {
                cluster_indices.push([idx])
            })

            cluster_indices.forEach(cluster => {
                let cluster_coor = cluster.map(i => {
                    return embeddedData[i].coor;
                })
                let rawContour = hull(cluster_coor, 1.5);
                let contour = rawContour.reduce((acc, d) => {
                    let contourMargin = 0.2;
                    let surrounders = [[d[0] + contourMargin, d[1] + contourMargin],
                                       [d[0] - contourMargin, d[1] + contourMargin],
                                       [d[0] + contourMargin, d[1] - contourMargin],
                                       [d[0] - contourMargin, d[1] - contourMargin]]
                    return acc.concat(surrounders);
                }, [])
                contour = hull(contour, 1.5);
                
                cluster_info.push({
                    indices: cluster,
                    contour: contour
                })
            })
            

            // svg.selectAll(".contour")
            //    .data(cluster_info.filter(d => { return d.contour.length > 3;}))
            //    .enter()
            //    .append("path")
            //    .attr("class", "contour")
            //    .attr("fill", "none")
            //    .attr("stroke-width", 1.5)
            //    .attr("stroke", "blue")
            //    .attr("d", datum => d3.line()
            //                     .x(d => xScale(d[0]))
            //                     .y(d => yScale(d[1]))
            //                     .curve(d3.curveCatmullRom)
            //                     (datum.contour)
            //    )


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

            console.log(edge_data)

          


            
            
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

