import React from 'react';
import { useEffect } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'


function FimifView(props) {

    useEffect(() => { 
        // Executes when new data arrives
        const jsonFileName = props.dataset + "_" + props.method + ".json";
        const data = require("../json/" + jsonFileName);

        const embeddedData = data.map(d => d.emb);
        
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

        const svg = d3.select("#scatterplot")
                      .attr("width", width + margin.hor * 2)
                      .attr("height", height + margin.ver * 2)
                        .append("g")
                        .attr("transform", "translate(" + margin.hor + ", " + margin.ver + ")");
        
        svg.selectAll("circle")
           .data(embeddedData)
           .join(
               enter => {
                   enter.append("circle")
                        .attr("fill", "blue")
                        .attr("cx", d => xScale(d[0]))
                        .attr("cy", d => yScale(d[1]))
                        .attr("r", radius);

               }
           )


    }, [props.dataset, props.method])


    return (
        <div style={{width: props.width * 1.1}}>
            <H5>{props.method.toUpperCase()} embedding result of {props.dataset.toUpperCase()} dataset</H5>
            <svg id="scatterplot"></svg>
        </div>
    )
}

const H5 = styled.h5`
    margin: 3px;
    font-size: 1.1em; 
    text-align: center;
`;

export default FimifView;

