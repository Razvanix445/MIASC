import axios from 'axios';

const API_URL = 'http://localhost:5000';

class PredictionService {
    uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        return axios.post(`${API_URL}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    }
}

export default new PredictionService();