import React from 'react';
import Header from "./js/Header"
import FimifView from "./js/FimifView"




function App() {

  const size = 500;


  return (
    <div>
      <Header/>
      <FimifView
        method="tsne"
        dataset="sphere"
        height={size}
        width={size}
      />
    </div>
  );
}

export default App;
