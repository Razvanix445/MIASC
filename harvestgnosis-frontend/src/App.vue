<template>
  <div id="app">
    <header>
      <h1>Pomegranate Disease Detector</h1>
    </header>

    <main>
      <div v-if="!predictionResult">
        <ImageUploader @prediction-result="handleResult" />
      </div>
      <div v-else>
        <PredictionResults
            :result="predictionResult"
            @reset="resetAnalysis"
        />
      </div>
    </main>

    <footer>
      <p>&copy; 2025 Pomegranate Disease Detector</p>
    </footer>
  </div>
</template>

<script>
import ImageUploader from './components/ImageUploader.vue';
import PredictionResults from './components/PredictionResults.vue';

export default {
  name: 'App',
  components: {
    ImageUploader,
    PredictionResults
  },
  data() {
    return {
      predictionResult: null
    };
  },
  methods: {
    handleResult(result) {
      this.predictionResult = result;
      console.log('Prediction received:', result);
    },
    resetAnalysis() {
      this.predictionResult = null;
    }
  }
}
</script>

<style>
body {
  margin: 0;
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background-color: #f9f9f9;
}

#app {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

header {
  text-align: center;
  margin-bottom: 30px;
}

header h1 {
  color: #42b983;
}

footer {
  text-align: center;
  margin-top: 40px;
  color: #6c757d;
  font-size: 14px;
}
</style>