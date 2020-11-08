import React from 'react';
import styled from 'styled-components'

import Header from "./components/Header"
import FimifMap from "./components/FimifMap"
import FimifMapFalse from "./components/FimifMapFalse"



function App() {

  const size = 800;

  return (
    <div>
      <Header/>
      <FimifWrapper>      
        {/* <FimifMap
          method="tsne"
          dataset="mnist_sampled_10"
          height={size}
          width={size}
          isLabel={false}
          metric = "euclidean"
        /> */}
        <FimifMapFalse
          method="tsne"
          dataset="mnist_sampled_10"
          height={size}
          width={size}
          isLabel={false}
          metric = "euclidean"
        />

      </FimifWrapper>
    </div>
  );
}

const FimifWrapper = styled.div`
  justify-content: center;
  display: flex;
  alignIterms: 'center';

`

export default App;
