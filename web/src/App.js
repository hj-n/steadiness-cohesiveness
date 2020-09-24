import React from 'react';
import AppBar from '@material-ui/core/AppBar'
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';


const useStyles = makeStyles({
  header : {
    background: "#8ae1f2", 
    padding: 10,
    color: "black"
    
  }
})


function App() {

  const classes = useStyles();

  return (
    <div>
      <AppBar className={classes.header}>
        <Typography variant="h6">
          FiMiF
        </Typography>
      </AppBar>
      
    </div>
  );
}

export default App;
