import React from 'react'
import './myStylesheet.css'

const nextLine = {
    fontSize: '72px',
    color: 'red'
}

function Food() {
    return (
        <div>
            <h1 className="primary">Cookin some chicken</h1>
            <h2 style={nextLine}>ahh</h2>
        </div>
    )
}

export default Food
