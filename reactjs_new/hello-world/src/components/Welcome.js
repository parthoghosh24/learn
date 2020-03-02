import React, { Component} from 'react'
// import React from 'react'

class Welcome extends Component {
   render(){
    return(
        <div>
            <h1>I am {this.props.superhero}</h1>
            {this.props.children}
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