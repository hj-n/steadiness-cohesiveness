import React from 'react';
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'


function FimifView(props) {

    let jsonFileName = props.dataset + "_" + props.method + ".json";
    let data = require("../json/" + jsonFileName);
    const embeddedData = data.map((d, i) => {
        if(d.emb.length === 2) d.emb.push(i);   // avoid double pushing (due to rendering issue)
        return d.emb;
    });

    // circle brusing
    const [cbx, setCbx]     = useState(0);
    const [cby, setCby]     = useState(0);
    const [bx, setBx]       = useState(0);     // current brushing position (x)
    const [by, setBy]       = useState(0);     // current brushing position (x)
    const [cbidx, setCbidx] = useState(0);     // circle index
    const [isCb, setIsCb]   = useState(false); // is brushing?

    const cbrush = useRef();


    useEffect(() => { 
        // Executes when new data arrives
        const width = props.width;
        const height = props.height;
        const margin = { hor: props.width / 20, ver: props.height / 20 };

        console.log(embeddedData);

        const [minX, maxX] = d3.extent(embeddedData, d => d[0]);
        const [minY, maxY] = d3.extent(embeddedData, d => d[1]);

        const xScale = d3.scaleLinear()
                         .domain([minX, maxX])
                         .range([0, width]);
        
        const yScale = d3.scaleLinear()
                         .domain([minY, maxY])
                         .range([0, height]);
        
        const radius = 3;

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
                        .attr("fill", "blue")
                        .attr("cx", d => xScale(d[0]))
                        .attr("cy", d => yScale(d[1]))
                        .style("opacity", 0.3)
                        .attr("r", radius);
               }
           );
        
        svg.selectAll("circle")
           .on("mouseover", function(){ d3.select(this).style("opacity", 1).attr("r", radius + 2); })
           .on("mouseout" , function(){ d3.select(this).style("opacity", 0.3).attr("r", radius);   })
           .call(d3.drag()
                   .on("start", function() { 
                       let circleData = d3.select(this).data()[0];
                       setIsCb(true); 
                       setCbidx(circleData[2]);
                       setCbx(xScale(circleData[0])); setBx(xScale(circleData[0]));
                       setCby(yScale(circleData[1])); setBy(yScale(circleData[1]));
                    })
                   .on("drag", (event) => {
                       setBx(event.x);
                       setBy(event.y);
                   })
                   .on("end", () => {
                       cbrush.current.style("opacity", 0.3);
                       setIsCb(false);
                   })
           
           )
        // ANCHOR brushing
        cbrush.current = svg.append("circle").style("opacity", 0);
    }, [])


    // For brushing
    useEffect(() => {
        if(isCb) {
            cbrush.current.style("opacity", 0.5)
                          .attr("cx", cbx)
                          .attr("cy", cby)
                          .attr("r", () => { return Math.sqrt(Math.pow((cbx - bx),2) + Math.pow((cby - by),2))} )
        }
    }, [by])


    return (
        <div style={{width: props.width * 1.1}}>
            <H5>{props.method.toUpperCase()} embedding result of {props.dataset.toUpperCase()} dataset</H5>
            <svg id={"scatterplot" + props.dataset + props.method}></svg>
        </div>
    )
}

const H5 = styled.h5`
    margin: 3px;
    font-size: 1.1em; 
    text-align: center;
`;

export default FimifView;

