<template>
  <div class="uploader-container">
    <h2>Upload Pomegranate Image</h2>

    <div class="upload-area" @click="triggerFileInput" @dragover.prevent @drop.prevent="onFileDrop">
      <input
          type="file"
          ref="fileInput"
          @change="onFileSelected"
          accept="image/jpeg,image/png,image/jpg"
          style="display: none"
      >
      <div v-if="!selectedFile" class="upload-prompt">
        <i class="upload-icon">📁</i>
        <p>Click to browse or drag and drop an image</p>
      </div>
      <div v-else class="file-preview">
        <img :src="filePreview" alt="Preview" class="preview-image">
        <p>{{ selectedFile.name }}</p>
      </div>
    </div>

    <button
        @click="uploadImage"
        :disabled="!selectedFile || isLoading"
        class="upload-button"
    >
      {{ isLoading ? 'Processing...' : 'Analyze Image' }}
    </button>

    <div v-if="isLoading" class="loader"></div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>
  </div>
</template>

<script>
import PredictionService from '@/services/PredictionService';

export default {
  name: 'ImageUploader',
  data() {
    return {
      selectedFile: null,
      filePreview: null,
      isLoading: false,
      error: null
    };
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    onFileSelected(event) {
      const file = event.target.files[0];
      if (file) {
        this.processFile(file);
      }
    },
    onFileDrop(event) {
      const file = event.dataTransfer.files[0];
      if (file && file.type.match('image.*')) {
        this.processFile(file);
      } else {
        this.error = 'Please drop an image file';
      }
    },
    processFile(file) {
      this.selectedFile = file;
      this.error = null;

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        this.filePreview = e.target.result;
      };
      reader.readAsDataURL(file);
    },
    async uploadImage() {
      if (!this.selectedFile) return;

      this.isLoading = true;
      this.error = null;

      try {
        const response = await PredictionService.uploadImage(this.selectedFile);
        this.$emit('prediction-result', response.data);
      } catch (err) {
        console.error('Upload error:', err);
        this.error = err.response?.data?.error || 'Failed to process image';
      } finally {
        this.isLoading = false;
      }
    }
  }
}
</script>

<style scoped>
.uploader-container {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
}

.upload-area {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  cursor: pointer;
  margin-bottom: 20px;
  transition: border-color 0.3s;
}

.upload-area:hover {
  border-color: #42b983;
}

.upload-icon {
  font-size: 48px;
  margin-bottom: 10px;
  display: block;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  margin-bottom: 10px;
  border-radius: 4px;
}

.upload-button {
  background-color: #42b983;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  width: 100%;
}

.upload-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.loader {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #42b983;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 2s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-message {
  color: #dc3545;
  margin-top: 10px;
  text-align: center;
}
</style>