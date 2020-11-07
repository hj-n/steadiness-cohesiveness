import React from 'react';
import AppBar from '@material-ui/core/AppBar'
import Typography from '@material-ui/core/Typography';
import Toolbar from '@material-ui/core/Toolbar'



function Header() {
    return (
      <div>
      <AppBar>
        <Toolbar>
          <Typography variant="h6">Fimif Visualization</Typography>
        </Toolbar>
      </AppBar>

      <Toolbar/> {/* for content space */}
      </div>
    )
}

export default Header;