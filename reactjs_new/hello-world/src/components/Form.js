import React, { Component } from 'react'

class Form extends Component {
    constructor(props) {
        super(props)
    
        this.state = {
             username: '',
             comments: '',
             color: 'green'
        }
    }

    handleUsername = event =>{
        this.setState({
            username: event.target.value
        })
    }

    handleComments = event => {
        this.setState({
            comments: event.target.value
        })
    }

    handleColor = event => {
        this.setState({
            color: event.target.value
        })
    }

    handleSubmit = event => {
        alert(`${this.state.username} and ${this.state.comments} and ${this.state.color}`);
        event.preventDefault()
    }
    
    render() {
        return (
            <form onSubmit={this.handleSubmit}>
                <div>
                    <label>Username</label>
                    <input type="text" value={this.state.username} onChange={this.handleUsername}/>
                </div>
                <div>
                    <label>Comments</label>
                    <textarea value={this.state.comments} onChange={this.handleComments}></textarea>
                </div>
                <div>
                    <label>Colors</label>
                    <select value={this.state.color} onChange={this.handleColor}>
                        <option value="red">Red</option>
                        <option value="green">Green</option>
                        <option value="blue">Blue</option>
                    </select>
                </div>
                <button type="submit">Submit</button>
            </form>
        )
    }
}

export default Form
