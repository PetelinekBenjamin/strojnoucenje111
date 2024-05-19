const express = require('express');
const request = require('request');

const app = express();
const port = 3000;

app.use(express.static('public'));

app.get('/getPredictions', (req, res) => {
    // Klic API-ja
    request('http://api:5000/predict/naloga02', (error, response, body) => {
        if (!error && response.statusCode == 200) {
            const data = JSON.parse(body);
            res.send(data.prediction);
        } else {
            res.status(500).send('Napaka pri pridobivanju podatkov');
        }
    });
});

app.listen(port, () => {
    console.log(`Strežnik je zagnan in posluša na portu ${port}`);
});
