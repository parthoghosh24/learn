import React from 'react'

// function Greet() {
//     return <h1>Hello Partho</h1>
// }

export const Greet = () => {
    //JSX implementation
    
    // return (
    //     <div className="massaman">
    //         <h1>Hello Partho</h1>
    //     </div>
    // )

    // Regular JS
    return React.createElement(
        'div',
        {id: 'hello', className: "massaman"},
        React.createElement('h1', null, 'Hello Partho')
    )
}

// export default Greet;  // default export 