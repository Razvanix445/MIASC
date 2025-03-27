<template>
  <div class="results-container" v-if="result">
    <h2>Prediction Results</h2>

    <div class="result-card">
      <div class="result-header">
        <h3>{{ formatDiseaseName(result.prediction.disease) }}</h3>
        <div class="confidence-badge">
          {{ result.prediction.confidence.toFixed(2) }}% Confidence
        </div>
      </div>

      <div class="result-details">
        <p v-if="result.prediction.disease === 'Healthy'">
          Good news! Your pomegranate appears to be healthy.
        </p>
        <p v-else>
          Your pomegranate may have {{ formatDiseaseName(result.prediction.disease) }}.
          Please take appropriate measures.
        </p>
      </div>

      <button @click="$emit('reset')" class="reset-button">
        Analyze Another Image
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PredictionResults',
  props: {
    result: Object
  },
  methods: {
    formatDiseaseName(name) {
      return name.replace(/_/g, ' ');
    }
  }
}
</script>

<style scoped>
.results-container {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
}

.result-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.result-header {
  background-color: #f8f9fa;
  padding: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #e0e0e0;
}

.result-header h3 {
  margin: 0;
  color: #333;
}

.confidence-badge {
  background-color: #42b983;
  color: white;
  padding: 5px 10px;
  border-radius: 20px;
  font-size: 14px;
}

.result-details {
  padding: 20px;
  font-size: 16px;
  line-height: 1.5;
}

.reset-button {
  background-color: #6c757d;
  color: white;
  border: none;
  padding: 10px 20px;
  width: 100%;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.reset-button:hover {
  background-color: #5a6268;
}
</style>