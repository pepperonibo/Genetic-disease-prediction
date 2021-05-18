import { useState } from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import styles from './IndexPage.css';
import ReturnIndex from '../components/ReturnIndex';

const { Dragger } = Upload;

const TrainPage = () => {
  const [result, setResutl] = useState(null)

  const props = {
    name: 'file',
    multiple: false,
    action: 'http://localhost:5000/train',
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
        setResutl(info.file.response.acc)
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };
  
  return (
    <div className={styles.normal}>
      <div className={styles.draggerDiv}>
        <Dragger {...props}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          {result ? <p className="ant-upload-text">训练准确率：{result * 100}%</p> : null}
          <p className="ant-upload-text">点击或拖拽包含遗传病人脸图片的压缩包进行训练</p>
          <p className="ant-upload-hint">
            请确保压缩包压缩格式为zip，并将压缩包名重命名为疾病名称。目前由于内置正常儿童人脸数据集为1000张，若训练集图片大于1000张需要手动拓展正常脸数据集
          </p>
        </Dragger>
      </div>
      <ReturnIndex />
    </div>
  );
}


export default TrainPage;

// ReactDOM.render(
//   <Dragger {...props}>
//     <p className="ant-upload-drag-icon">
//       <InboxOutlined />
//     </p>
//     <p className="ant-upload-text">点击或拖拽包含遗传病人脸图片的压缩包进行训练</p>
//     <p className="ant-upload-hint">
//       请确保压缩包压缩格式为zip，并将压缩包名重命名为疾病名称。目前由于内置正常儿童人脸数据集为1000张，若训练集图片大于1000张需要手动拓展正常脸数据集
//     </p>
//   </Dragger>,
//   mountNode,
// );