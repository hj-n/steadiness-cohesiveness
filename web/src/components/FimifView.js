import React from 'react';
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3'
import styled from 'styled-components'
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';


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

    const [rMode, setRMode] = useState("2d") // true: use max 2D dist point, false: use max ND dist point
    const handleChange = (event) => {
        setRMode(event.target.value);
    };

    const cbrush = useRef();

    const xS = useRef();   // xScale
    const yS = useRef();   // yScale
    const cS = useRef();   // colorScale


    useEffect(() => { 
        // Executes when new data arrives
        const width = props.width;
        const height = props.height;
        const margin = { hor: props.width / 20, ver: props.height / 20 };

        const [minX, maxX] = d3.extent(embeddedData, d => d[0]);
        const [minY, maxY] = d3.extent(embeddedData, d => d[1]);

        const xScale = d3.scaleLinear()
                         .domain([minX, maxX])
                         .range([0, width]);
        
        const yScale = d3.scaleLinear()
                         .domain([minY, maxY])
                         .range([0, height]);

        const colorDomain = [...Array(props.labelNum).keys()]
        console.log(colorDomain);
        const colorRange = ["#4e79a7","#f28e2c","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab", "#e15759","#5E4FA2"];
        // const colorRange = ["#3288BD", "#5E4FA2", "#66C2A5", "#ABDDA4", "#E6F598", "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"];
        const colorScale =  d3.scaleOrdinal()
            .domain(colorDomain)
            .range(colorRange);
        cS.current = colorScale;

        xS.current = xScale;
        yS.current = yScale;
        
        
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
                        .attr("fill", d => {
                            if(props.isLabel) {
                                return colorScale(data[d[2]].label);
                            }   
                            else return "blue";
                        })
                        .attr("cx", d => xScale(d[0]))
                        .attr("cy", d => yScale(d[1]))
                        .style("opacity", 0.8)
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
           
           );

           svg.on("click", () => {
               svg.selectAll("circle")
                  .style("opacity", 0.3)
                  .attr("fill", d => {
                      if(d === undefined) return "blue";
                      if(props.isLabel) {
                          return colorScale(data[d[2]].label);
                      }   
                      else return "blue";
                  });
               cbrush.current.style("opacity", 0);
           });
           d3.select("#scatterplot" + props.dataset + props.method).on("click", () => {
                svg.selectAll("circle")
                   .style("opacity", 0.3)
                   .attr("fill", d => {
                       if(d === undefined) return "blue";
                       if(props.isLabel) {
                           return colorScale(data[d[2]].label);
                       }   
                       else return "blue";
                   });
                cbrush.current.style("opacity", 0).attr("r", 0);
           });
        
        
        // ANCHOR brushing
        cbrush.current = svg.append("circle").style("opacity", 0);
    }, [])


    // Called while brushing
    useEffect(() => {
        if(isCb) {
            const curR = Math.sqrt(Math.pow((cbx - bx),2) + Math.pow((cby - by),2))
            cbrush.current.style("opacity", 0.5)
                          .attr("cx", cbx)
                          .attr("cy", cby)
                          .attr("r", curR);
        }
    }, [by])

    // Calls after brushing
    useEffect(() => {
        if(!isCb && cbx !== bx) {
            
            // Render 2d-circle-contained point
            let svg = d3.select("#scatterplot" + props.dataset + props.method);


            function NDEuclideanDistance(arr1, arr2) {
                if(arr1.length !== arr2.length) throw "Array length Mismatch";
                let sum = 0;
                for(let i = 0; i < arr1.length; i++) 
                    sum += Math.pow(arr1[i] - arr2[i], 2);
                return Math.sqrt(sum);
            }

            function NDCosineDistance(arr1, arr2) {
                if(arr1.length !== arr2.length) throw "Array length Mismatch";
                let dot = 0;
                let m1  = 0;
                let m2  = 0;
                for(let i = 0; i < arr1.length; i++) {
                    dot += arr1[i] * arr2[i];
                    m1  += arr1[i] * arr1[i];
                    m2  += arr2[i] * arr2[i];
                }
                m1 = Math.sqrt(m1);
                m2 = Math.sqrt(m2);
                return 1 - dot / (m1 * m2);
            }

            const NDDistance = props.metric == "euclidean" ? NDEuclideanDistance : NDCosineDistance;
            
            let farthestIdx = 0;
            let farthestRadius = 0;
            svg.selectAll("circle")
               .style("opacity", (d) => {
                   if(d === undefined) return 0.3
                   const dist = Math.sqrt(Math.pow((cbx - xS.current(d[0])),2) + Math.pow((cby - yS.current(d[1])),2));
                   const curR =  Math.sqrt(Math.pow((cbx - bx),2) + Math.pow((cby - by),2))
                
                
                   // return opacity  
                   if(curR < dist) return 0.3;
                   else {
                       if(rMode === "2d") {
                            // update 2d-farthest point
                            if(dist > farthestRadius) {
                                farthestRadius = dist;
                                farthestIdx = d[2];
                            }
                        }
                       // update ND-farthest point
                       else {
                        let NDDist = NDDistance(data[d[2]].raw, data[cbidx].raw);
                            if(NDDist > farthestRadius) {
                                farthestRadius = NDDist;
                                farthestIdx = d[2];
                            }
                        }
                       return 0.8;
                   }
            });
            
            
            

            const farthestNDRadius = NDDistance(data[cbidx].raw, data[farthestIdx].raw);


            // find contained point in ND space
            let NDContainedIdices = {}
            for(let i = 0; i < data.length; i++) {
                let curNDRadius = NDDistance(data[cbidx].raw, data[i].raw);
                if(curNDRadius <= farthestNDRadius) 
                    NDContainedIdices[i] = true;
                else NDContainedIdices[i] = false;
            }

            svg.selectAll("circle")
               .attr("fill", d => {
                   if(d === undefined) return "blue";
                   if(NDContainedIdices[d[2]]) return "red";
                   else if(props.isLabel) {
                        return cS.current(data[d[2]].label);
                   }   
                   else return "blue";
               })
               .style("opacity", d => {
                   if(d === undefined) return "blue";
                   if(NDContainedIdices[d[2]]) return 1;
                   else return 0.3;

               })
        }
    }, [isCb, rMode])

    return (
        <div style={{width: props.width * 1.1, margin: 40}}>
            <H5>{props.dataset.toUpperCase()} dataset</H5>
            {/* <RadioWrapper>
                <RadioGroup row aria-label="gender" name="gender1" value={rMode} onChange={handleChange}>
                    <FormControlLabel value="2d" control={<Radio />} label="2D radius" />
                    <FormControlLabel value="nd" control={<Radio />} label="ND radius" />
                </RadioGroup>
            </RadioWrapper> */}
            <svg id={"scatterplot" + props.dataset + props.method}></svg>
        </div>
    )
}

const H5 = styled.h5`
    margin: 3px;
    font-size: 1.1em; 
    text-align: center;
`;

const RadioWrapper = styled.div`
    justify-content: center;
    display: flex;
    alignIterms: 'center';

`

export default FimifView;

