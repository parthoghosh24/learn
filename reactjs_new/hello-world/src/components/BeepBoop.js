import React, { Component } from 'react'

class BeepBoop extends Component {
    constructor(props) {
        super(props)
    
        this.state = {
             isBeeping: false
        }
    }
    
    render() {
        let message
        if(this.state.isBeeping)
        {
            message = <h1>Beep</h1>
        }
        else
        {
            message = <h1>Boop</h1>
        }

        return <div>{message}</div>
        // if(this.state.isBeeping)
        // {
        //     return (
        //         <div>
        //           <h1>Beep</h1>
        //         </div>
        //     )
        // }
        // else
        // {
        //     return (
        //         <div>
        //             <h1>Boop</h1>
        //         </div>
        //     )

        // }
        
    }
}

export default BeepBoop
