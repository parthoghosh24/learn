import React, { Component } from 'react'
import ChildComponent from './ChildComponent'

class ParentComponent extends Component {
    constructor(props) {
        super(props)
    
        this.state = {
             message: "I am parent"
        }
    }
    changeHandler=()=>{
        this.setState({
            message: "I am the child"
        })
    }
    
    render() {
        return (
            <div>
                <p>{this.state.message}</p>
                <ChildComponent changeHandler = {this.changeHandler}/>
            </div>
        )
    }
}

export default ParentComponent
