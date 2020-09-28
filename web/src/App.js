import React from 'react';
import styled from 'styled-components'

import Header from "./components/Header"
import FimifView from "./components/FimifView"



function App() {

  const size = 1000;


  return (
    <div>
      <Header/>
      {/* <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="sphere"
          height={size}
          width={size}
          isLabel={false}
          metric = "euclidean"
        />
        <FimifView
          method="tsne"
          dataset="swiss_roll"
          height={size}
          width={size}
          isLabel={false}
          metric = "euclidean"
        />
      </FimifWrapper>
      <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="mnist_test_euclidean"
          height={size}
          width= {size}
          isLabel={true}
          labelNum = {10}
          metric = "euclidean"
        />
      </FimifWrapper>
      <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="mnist_test_cosine_similarity"
          height={size}
          width= {size}
          isLabel={true}
          labelNum = {10}
          metric = "cosine_similarity"
        />
      </FimifWrapper> */}
      <FimifWrapper>
        <FimifView
          method="umato"
          dataset="spheres"
          height={size * 0.5}
          width= {size * 0.5}
          isLabel={true}
          labelNum = {10}
          metric = "euclidean"
        />
        <FimifView
          method="umap"
          dataset="spheres"
          height={size * 0.5}
          width= {size * 0.5}
          isLabel={true}
          labelNum = {10}
          metric = "euclidean"
        />
      </FimifWrapper>
      <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="spheres"
          height={size * 0.5}
          width= {size * 0.5}
          isLabel={true}
          labelNum = {10}
          metric = "euclidean"
        />
        <FimifView
          method="pca"
          dataset="spheres"
          height={size * 0.5}
          width= {size * 0.5}
          isLabel={true}
          labelNum = {10}
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
