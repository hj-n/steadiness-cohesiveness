import React from 'react';
import Header from "./components/Header"
import FimifView from "./components/FimifView"




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
