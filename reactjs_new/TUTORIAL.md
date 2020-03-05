ReactJS
===============

# Basics

* It follows component based architecture (In a website, components can be header,footer, sidenav and main)
* React is declarative
* React can be seamlessly integrated in application. It can be a part of a page to full blown application.
* Prequisites to learn React:
  - In javascript-> this, filter, map and reduce are important
  - In ES6-> let & const, arrow functions, template literals, default params, object literals, rest and spread operators, destructuring assignment
  - HTML & CSS
* React learn list:
  - Fundamentals
  - HTTP
  - Routing
  - Redux
  - Utilties

## React folder structure:

A typical reactjs project has following folder structure:

node modules (dir)
-------------  
This contains all the downloaded libraries mentioned in package.json. This generally gets populated after 'npm install'


public (dir)
--------------
Contains index.html and all static assets


src (dir)
--------------
Contains all source code (JS, css, etc)


package.json
---------------
Used for dependency management. Something similar to Gemfile in Rails.


yarn.lock/package.lock
-----------------------
Depedency management version file


## Component based architecture

A react app will generally have two kinds of components:

* Functional component
  - Javascript functions
  - Dumb components
  - 'this' keyword
  -  We can use '<i>hooks</i>' to implement state in function

* Class component
  - class extends component class
  - Render method returning html
  - Can contain private state
  - For complex UI logic
  - Provide lifecycle hooks
  - Also known as Stateful/ Smart/ Container

Check Greet.js from more info

## JSX

* Extension to the Javascript language syntax.
* Write XML-like code for elements and components.
* JSX tags have a tag name, attributes & children.
* JSX is not a necessity to write React applications.
* Transpiles to pure javascript which is understood by the browsers.
* its elegant

## Props

* Brings dynamic behavior to the html elements.
* We can pass parameters to JSX tags
* Get passed to the component
* Function parameters
* props are immutable
* used as 'props' in Functional Components & 'this.props' in Class Components

## State

* Is managed within the component
* Variables declared in the function body
* state can be changed (mutable)
* useState Hook in functional components & this.state in Class Components
* Check <b>Message.js</b> in components
* setState - Helps in mutating state
* setState - its asynchronous
* setState - prevState can be used to access prev state
* eventBinding - Please refer <b>Subscribe.js</b>

## Parent child communication in components

* We can send Child data to parents via props. Refer <b>ParentComponent.js</b> and <b>ChildComponent.js</b> 

## Conditional rendering

* Four ways to achieve this:
 - If else
 - environmental variables using let
 - ternary operator
 - &&

 ## List rendering

* We achieve this via 'map' function in javascript.
* Standard is to define a common component for each list item
* key plays an important role react lists as it helps react render and update list efficiently
* Avoid using list index as key


