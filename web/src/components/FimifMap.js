import React from 'react';
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import { orange } from '@material-ui/core/colors';


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

    const xS = useRef();   // xScale
    const yS = useRef();   // yScale
    const cS = useRef();   // colorScale
    
    const threshold = useRef();    // threshold for determining the saturation of false score

    useEffect(() => {
        const width = props.width;
        const height = props.height;
        const margin = { hor: props.width / 20, ver: props.height / 20 };

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
        
        
        threshold.current = 0.6;   // will be changed (with slidebar interaction)

        let powList = embeddedData.map(d => {
            return Math.pow(Math.pow(d.false_value[0], 2) + Math.pow(d.false_value[1], 2), 0.5)
        });
        powList.sort()
        console.log(powList)
        let thresholdIdx = Math.round(powList.length * threshold.current)
        let maxVal = powList[powList.length - 1]
        let thresholdVal = powList[thresholdIdx]
        let domain = [maxVal - 2 * (maxVal - thresholdVal), maxVal]
        // let domain = [domain[1] - 2*(domain[1] - threshold.current), domain[1]]
        console.log(domain)

        let colorScale = d3.scaleSequential().domain(domain).interpolator(d3.interpolateGreys);

        cS.current = colorScale;
        xS.current = xScale;
        yS.current = yScale;


        
        const radius = 3.5;

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
                            let power = Math.pow(Math.pow(d.false_value[0], 2) + Math.pow(d.false_value[1], 2), 0.5);
                            if (power < thresholdVal) power = thresholdVal
                            return colorScale(power)
                        })
                        .attr("cx", d => xScale(d.coor[0]))
                        .attr("cy", d => yScale(d.coor[1]))
                        .style("opacity", 0.8)
                        .attr("r", radius)
               }
           );
        
        svg.selectAll("circle")
           .on("mouseover", function(){ d3.select(this).style("opacity", 1).attr("r", radius * 2); })
           .on("mouseout" , function(){ d3.select(this).style("opacity", 0.5).attr("r", radius);   })
        
        

        

        

    }, [])




    return (
        <div style={{width: props.width * 1.1, margin: 5}}>
            <H6>{props.dataset} dataset embedded by {props.method}</H6>
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

