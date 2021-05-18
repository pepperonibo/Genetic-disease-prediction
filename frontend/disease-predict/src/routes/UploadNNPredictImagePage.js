import React from 'react';
import styles from './IndexPage.css';
import { Upload, message } from 'antd';
import { LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import ReturnIndex from '../components/ReturnIndex';

function getBase64(img, callback) {
  const reader = new FileReader();
  reader.addEventListener('load', () => callback(reader.result));
  reader.readAsDataURL(img);
}

function beforeUpload(file) {
  const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
  if (!isJpgOrPng) {
    message.error('You can only upload JPG/PNG file!');
  }
  const isLt2M = file.size / 1024 / 1024 < 2;
  if (!isLt2M) {
    message.error('Image must smaller than 2MB!');
  }
  return isJpgOrPng && isLt2M;
}

class Avatar extends React.Component {
  state = {
    loading: false,
    predict_result: null
  };

  handleChange = info => {
    if (info.file.status === 'uploading') {
      this.setState({ loading: true });
      return;
    }
    if (info.file.status === 'done') {
      console.log(info.file.response.resutl)
      // Get this url from response in real world.
      getBase64(info.file.originFileObj, imageUrl =>
        this.setState({
          imageUrl,
          loading: false,
          predict_result: info.file.response.resutl
        }),
      );
    }
  };

  render() {
    const { loading, imageUrl } = this.state;
    const uploadButton = (
      <div>
        <div>
          {loading ? <LoadingOutlined /> : <PlusOutlined />}
          <div style={{ marginTop: 8 }}>Upload</div>
        </div>
      </div>
    );
    return (
      <div>
        <Upload
          name="file"
          listType="picture-card"
          className="avatar-uploader"
          showUploadList={false}
          action="http://localhost:5000/predict_nn"
          beforeUpload={beforeUpload}
          onChange={this.handleChange}
          data={{'disease': this.props.disease}}
        >
          {imageUrl ? <img src={imageUrl} alt="avatar" style={{ width: '100%' }} /> : uploadButton}
        </Upload>
        <div className={styles.predictResult}>{this.state.predict_result ? "预测结果：" + this.state.predict_result : null}</div>
      </div>
    );
  }
}

class UploadPredictImagePage extends React.Component {
  render() {
      return (
        <div className={styles.normal}>
          <h1 className={styles.title}>请选择需要预估患病概率的图片</h1>
          <Avatar disease={this.props.location.state.disease}/>
          <ReturnIndex/>
        </div>
      )
  }
}

export default UploadPredictImagePage;