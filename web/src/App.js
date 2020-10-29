import React from 'react';
import styled from 'styled-components'

import Header from "./components/Header"
import FimifView from "./components/FimifView"
import FimifMap from "./components/FimifMap"



function App() {

  const size = 300;

  return (
    <div>
      <Header/>
      
      <FimifWrapper>
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_0"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_1"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_3"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_4"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_5"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />


      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_6"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
        </FimifWrapper>
<FimifWrapper>
        <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_7"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_8"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />



      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_9"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_10"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
       
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_11"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_12"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      <FimifView
          method="none"
          dataset="multiclass_swissroll_oneside_13"
          height={size}
          width={size}
          isLabel={true}
          metric = "euclidean"
        />
      
        </FimifWrapper>

      {/* <FimifMap
        method="pca"
        dataset="swiss_roll"
        height={size}
        width={size}
        isLabel={false}
      /> */}
      {/* <FimifMap
        method="tsne"
        dataset="swiss_roll"
        height={size}
        width={size}
        isLabel={false}
      /> */}
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
      </FimifWrapper>
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
          method="topoae"
          dataset="spheres"
          height={size * 0.5}
          width= {size * 0.5}
          isLabel={true}
          labelNum = {10}
          metric = "euclidean"
        />
      </FimifWrapper> */}
      
    </div>
  );
}

const FimifWrapper = styled.div`
  justify-content: center;
  display: flex;
  alignIterms: 'center';

`

export default App;
