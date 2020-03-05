import React, { Component } from 'react'
import ListItem from './ListItem'

class List extends Component {

    render() {
        const cars = [
            {id: 1, name: "MG Hector", type: "SUV", img: "http://informus.ru/wp-content/uploads/2020/02/12-400x400.jpg"},
            {id: 2, name: "Honda City", type: "Sedan", img: "https://winmin.in/wp-content/uploads/2018/12/9_Orchid-White-Pearl-400x400.jpg"},
            {id: 3, name: "Maruti Baleno", type: "Hatchback", img: "https://winmin.in/wp-content/uploads/2018/12/maruti-baleno-posture-1-400x400.jpg"}
        ]
        const carList = cars.map(car => <ListItem key={car.id} carObject = {car}/>)
        return  <div> {carList} </div>
        
    }
}

export default List
