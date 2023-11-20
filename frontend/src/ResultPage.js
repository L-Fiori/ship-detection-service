import React, { useEffect, useState } from 'react';
import './ResultPage.css';

function ResultPage() {
  const [names, setNames] = useState([])
  const [images, setImages] = useState([])
  const [products, setProducts] = useState([])
  const [ships, setShips] = useState([])
  const [properties, setProperties] = useState([])

  useEffect(() => {
    fetch("http://localhost:8080/result_page")
    .then((response) => response.json())
    .then((data) => {
      setNames(data.names)
      setImages(data.images)
      setProducts(data.products)
      setShips(data.ships)
      setProperties(data.properties)
      })
  }, [])


  return (
    <div className="results-container">
      <div class="wrapper">
      <h2 className="results-message">Results:</h2>
      {products.map((product, index_product) => (
        <div class={"collapsible product "+index_product}>
          <input type="checkbox" id={"collapsible-head product "+index_product}/>
          <label for={"collapsible-head product "+index_product}>{"Product: " + product}</label>
          <div class={"collapsible-text product "+index_product}>
            {images[index_product].map((image, index_img) => (

            <div class={"collapsible image "+index_product+index_img}>
              <input type="checkbox" id={"collapsible-head image "+index_product+index_img}/>
              <label for={"collapsible-head image "+index_product+index_img}>

              <div class={"collapsible-row "+index_product+index_img}>
                <img class={"collapsible-item img "+index_product+index_img} key={product + " " + index_product+index_img} src={"data:image/png;base64, "+image} style={{width: '150px'}}/>
                <div class={"collapsible-item name "+index_product+index_img}>Image name: {names[index_product][index_img]}</div>
                <div class={"collapsible-item ship "+index_product+index_img}>Ship count: {ships[index_product][index_img]}</div>
              </div>

              </label>
              <div class={"collapsible-text image "+index_product+index_img}>
                {properties[index_product][index_img].map((property_list, index_p) => (
                  <div class="collapsible-row-property">
                    <div class="collapsible-item-property">{"Ship number: " + property_list[0]}</div>
                    <div class="collapsible-item-property">{"Ship classification: " + property_list[1]}</div>
                    <div class="collapsible-item-property">{"Ship size: " + property_list[2]}</div>
                  </div>
                ))}
              </div>
            </div>

            ))}
          </div>

        </div>
      ))}
      </div>

    </div>
  )
}

export default ResultPage;
