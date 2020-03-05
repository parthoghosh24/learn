import React from 'react'

function ListItem({carObject}) {
    return (
        <div>
            <p>{carObject.name}</p>
            <p>{carObject.type}</p>
            <img src={carObject.img} alt={carObject.name}/>
        </div>
    )
}

export default ListItem