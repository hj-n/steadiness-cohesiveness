import React from 'react';
import { useRef, useEffect } from 'react'
import * as d3 from 'd3';
import { path } from 'd3';

function FimifMap(props) {


    let jsonFileName = props.dataset + "_" + props.method + ".json";
    let data = require("../json/" + jsonFileName);
    const embeddedData = data.map((d, i) => {
        if(d.emb.length === 2) d.emb.push(i);   // avoid double pushing (due to rendering issue)
        return d.emb;
    });

    let pathFileName = props.dataset + "_" + props.method + "_path.json";
    console.log(jsonFileName)
    console.log(pathFileName)
    let pathData = require("../json/" + pathFileName);
    console.log(pathData)

    const xS = useRef();   // xScale
    const yS = useRef();   // yScale
    const cS = useRef();   // colorScale

    useEffect(() => { 

        const width = props.width;
        const height = props.height;
        const margin = { hor: props.width / 20, ver: props.height / 20 };

        let [minX, maxX] = d3.extent(embeddedData, d => d[0]);
        let [minY, maxY] = d3.extent(embeddedData, d => d[1]);

        let x = 1.25

        minX = x * minX;
        maxX = x * maxX;
        minY = x * minY;
        maxY = x * maxY;


        const xScale = d3.scaleLinear()
                         .domain([minX, maxX])
                         .range([0, width]);
        
        const yScale = d3.scaleLinear()
                         .domain([minY, maxY])
                         .range([0, height]);

        const colorDomain = [...Array(props.labelNum).keys()]
        const colorRange = ["#4e79a7","#f28e2c","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab", "#e15759","#5E4FA2"];
        const colorScale =  d3.scaleOrdinal()
            .domain(colorDomain)
            .range(colorRange);
        cS.current = colorScale;

        xS.current = xScale;
        yS.current = yScale;
        
        
        const radius = 1.75;

        // ANCHOR SVG width / height setting
        const svg = d3.select("#scatterplot" + props.dataset + props.method)
                      .attr("width", width + margin.hor * 2)
                      .attr("height", height + margin.ver * 2)
                        .append("g")
                        .attr("transform", "translate(" + margin.hor + ", " + margin.ver + ")");
        
        // ANCHOR Scatterplot rending & interaction
        svg.selectAll("circle")
           .data(embeddedData)
           .join(
               enter => {
                   enter.append("circle")
                        .attr("fill", d => {
                            if(props.isLabel) {
                                return colorScale(data[d[2]].label);
                            }   
                            else return "blue";
                        })
                        .attr("cx", d => xScale(d[0]))
                        .attr("cy", d => yScale(d[1]))
                        .style("opacity", 1)
                        .attr("r", radius);
               }
           );
        
        // svg.append("path")
        //    .data([points])
        //    .attr("d", d3.svg.line()
        //    .tension(0) // Catmull–Rom
        //    .interpolate("cardinal-closed"));
        // let line = d3.line()
        //             .x(d => d.x) // set the x values for the line generator
        //             .y(d => d.y) // set the y values for the line generator 
        //             .curve(d3.curveMonotoneX) // apply smoothing to the line

        let line = d3.line(d => xScale(d.x), d => yScale(d.y)).curve(d3.curveMonotoneX)
                                                             
                     
        
        // console.log(pathData[0])
        // // for (let i = 0; i < pathData.length; i++) {
        //     // console.log(pathData[i])
        //     console.log(pathData)

        for (let i = 0; i < pathData.length; i++) {
            let test = pathData[i].map(d => {
                return {
                    x: d[0],
                    y: d[1]
                }
            });
            // console.log(test)

            let length = 1000

            svg.append("g")
               .selectAll("path")
               .data(test.slice(0,length))
               .join(
                   enter => enter.
                    append("path")
                    .attr("d", (d,i,nodes) => {
                        return line([test[i], test[i+1]])
                    })
                    .attr("stroke", (d) => {
                        if(props.isLabel) {
                            return colorScale(data[embeddedData[i][2]].label);
                        }   
                        else return "blue";
                    })
                    .attr("stroke-width", (d, i) => {
                        return 3 - 3 * (i / length);
                    })
                    .attr("fill", "none")
                    .style("opacity", (d, i) => {
                        return 1 - (i / length);
                    })
               )

            // svg
            //    .append("path")
            //    .attr("d", line(test.slice(0,1000)))
            //    .attr("stroke", "blue")
            //    .attr("stroke-width", 1)
            //    .attr("fill", "none")
            //    .style("opacity", 0.2)
            //    .style("opacity", (d,i) => {
            //        console.log(d)
            //        return 1 - (i / 100);
            //    });

            //    svg.append("path")
            //    .attr("d", line(test.slice(40, 100)))
            //    .attr("stroke", "blue")
            //    .attr("stroke-width", 1)
            //    .attr("fill", "none")
            //    .style("opacity", 0.3);
               
        }
    });

    return (
        <div style={{width: props.width * 1.1, margin: 40}}>
            <svg id={"scatterplot" + props.dataset + props.method}></svg>
        </div>
    )

}

export default FimifMap;