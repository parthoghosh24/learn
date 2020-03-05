import React, { Component } from 'react';
import './App.css';
import { Greet } from './components/Greet';
import Welcome from './components/Welcome';
import Message from './components/Message';
import Counter from './components/Counter';
import Subscribe from './components/Subscribe';
import ParentComponent from './components/ParentComponent';
import BeepBoop from './components/BeepBoop';
import List from './components/List';

class App extends Component{
  render()
  {
    return (
      <div className="App">
        <List/>
        {/* <BeepBoop/> */}
        {/* <ParentComponent/> */}
        {/* <Subscribe/> */}
        {/* <Counter/> */}
        {/* <Message/> */}
        {/* <Greet/>
        <Welcome superhero="Batman">
          <p>I am the night</p>
        </Welcome>
        <Welcome superhero="Ironman">
          <button>Activate</button>
        </Welcome>
        <Welcome superhero="One punch man">
          <h1>KO!!!!</h1>
        </Welcome> */}
      </div>
    );
  }
}

export default App;
