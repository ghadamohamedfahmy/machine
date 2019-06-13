using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.Api;
using Model;
using Microsoft.AspNetCore.Hosting;
using System.Web.Http.Description;

namespace TestApiCore.Controllers
{
    [Route("api/[controller]")]
    public class ValuesController : Controller
    {
        private readonly IHostingEnvironment _hostingEnvironment;
        PredictionModel<SentimentData, SentimentPrediction> model ;

        public ValuesController(IHostingEnvironment hostingEnvironment)
        {
            _hostingEnvironment = hostingEnvironment;
            readModel();
           
        }
        async void readModel()
        {
            try
            {
                model = await PredictionModel<SentimentData, SentimentPrediction>.ReadAsync<SentimentData, SentimentPrediction>(_hostingEnvironment.WebRootPath + "/Model.zip");
            }
            catch (Exception ex)
            {

            }

        }
        // GET api/values
        [HttpGet]
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/values/5
        [ResponseType(typeof (Rate))]
        [HttpGet("{text}")]
        public string Get(string text)
        {

            //prediction based on comments
            IEnumerable<SentimentData> sentiments = new[]

            {
                new SentimentData
                 {
                    SentimentText = text//"licencié et a quité l'entreprise. Elle formule une demande de formation dans le cadre du DIF. "

                 }/*,
                 new SentimentData
                 {
                    SentimentText = "Est ce qu une absence pour accident du travail a un impact sur l attribution des RTTCONGES ANNUELS / RTT / COMPTE EPARGNE TEMPSb"

                 },
                 new SentimentData
                 {
                    SentimentText = "New hr is on board"

                 },
                new SentimentData
                {
                    SentimentText = "i want go for vacation of 2 weeks"

                },
                 new SentimentData
                {
                    SentimentText = "comment enregistrer une adresse où apparait ancien chemin"

                },
                 new SentimentData
                {
                     SentimentText = " travail pour les auxiliaires de vacances ?   "

                }*/
            };
            //sentiment prediction
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            string x= "Sentiment Predictions\r\n";
            x+="---------------------\r\n";
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                x+=$"SentimentText: {item.sentiment.SentimentText} | Category: {(item.prediction.Category)} \r\n";
            }
            return x;
        }
        [HttpGet("myFunc/{text}")]
        public string MyFunc(string text )
        {
            return "Your Text : a :" + text;
        }
        // POST api/values
        [HttpPost]
        public void Post([FromBody]string value)
        {
        }

        // PUT api/values/5
        [HttpPut("{id}")]
        public void Put(int id, [FromBody]string value)
        {
        }

        // DELETE api/values/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
        }
    }
}
