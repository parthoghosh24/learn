import React, { Component} from 'react'
// import React from 'react'

class Welcome extends Component {
   render(){
       const {superhero, children} = this.props
    return(
        <div>
            <h1>I am {superhero}</h1>
            {children}
        </div>
    )
   }
    
}
    // render()
    // {
    //     return <h1>I am ReactJS</h1>
    // }
// }

export default Welcome