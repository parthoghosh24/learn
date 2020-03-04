import React, { Component } from 'react'

class Subscribe extends Component {
    constructor(props)
    {
        super(props)
        this.state ={
            message: "Click to subscribe"
        }
    }
    // Event binding
    subscribeClick=()=>{
        this.setState({
            message: "Thanks for subscribing"
        })
    }

    render() {
        return (
            <div>
                <h1>{this.state.message}</h1>
                <button onClick={this.subscribeClick}>Subscribe</button>
            </div>
        )
    }
}

export default Subscribe
