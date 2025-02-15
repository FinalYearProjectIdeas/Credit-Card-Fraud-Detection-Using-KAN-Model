document.addEventListener('DOMContentLoaded', function () {
  // Handle form submission
  document
    .getElementById('predictionForm')
    .addEventListener('submit', function (event) {
      event.preventDefault()

      let jsonInput = document.getElementById('jsonInput').value.trim()

      try {
        let jsonData = JSON.parse(jsonInput)

        if (
          !jsonData.features ||
          !Array.isArray(jsonData.features) ||
          jsonData.features.length !== 30
        ) {
          alert(
            'Invalid input! Please enter exactly 30 feature values in JSON format.'
          )
          return
        }

        // Send the request to the Flask backend
        fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(jsonData),
        })
          .then((response) => response.json())
          .then((data) => {
            let resultDiv = document.getElementById('predictionResult')
            if (data.error) {
              resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}`
              resultDiv.classList.add('alert-danger')
              resultDiv.classList.remove('alert-info')
            } else {
              resultDiv.innerHTML = `<strong>Fraud Probability:</strong> ${data.fraud_probability.toFixed(
                4
              )}`
              resultDiv.classList.add('alert-info')
              resultDiv.classList.remove('alert-danger')
            }
            resultDiv.classList.remove('d-none')
          })
          .catch((error) => {
            console.error('Error:', error)
            alert('Something went wrong! Please try again.')
          })
      } catch (e) {
        alert('Invalid JSON format! Please check your input.')
      }
    })
})
