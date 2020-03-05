import React from 'react'

function ChildComponent(props) {
    return (
        <div>
            <button onClick = {props.changeHandler}>ChangeParent</button>
        </div>
            
    )
}

export default ChildComponent
