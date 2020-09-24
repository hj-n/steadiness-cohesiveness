import React from 'react';
import styled from 'styled-components'

import Header from "./components/Header"
import FimifView from "./components/FimifView"



function App() {

  const size = 500;


  return (
    <div>
      <Header/>
      <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="sphere"
          height={size}
          width={size}
        />
        <FimifView
          method="tsne"
          dataset="swiss_roll"
          height={size}
          width={size}
        />
      </FimifWrapper>
      <FimifWrapper>
        <FimifView
          method="tsne"
          dataset="mnist_test_euclidean"
          height={size * 2}
          width= {size * 2}
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
