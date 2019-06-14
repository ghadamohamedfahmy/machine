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
using Newtonsoft.Json;
using System.Xml;

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
        [ResponseType(typeof (JsonResult))]
        [HttpGet]
        public IEnumerable<string> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/values/5
       
        [HttpGet("{text}")]
       
        public List<Rate> Get(string text)
        {

            //prediction based on comments
            IEnumerable<SentimentData> sentiments = new[]

            {
                new SentimentData
                 {
                    SentimentText = text                 }
            };
            //sentiment prediction
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            List<Rate> subjects = new List<Rate>();
            Rate rate=new Rate();
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            sentimentsAndPredictions.ToArray();
            foreach (var item in sentimentsAndPredictions)
            {



                rate.Category =  $"{ item.prediction.Category}";

                rate.SentimentText = $"{item.sentiment.SentimentText}";
                subjects.Add(rate);
            }
            //JsonConvert.SerializeObject(rate);
            //JsonConvert.DeserializeObject<Rate>(rate.SentimentText);

            return subjects;
           

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
