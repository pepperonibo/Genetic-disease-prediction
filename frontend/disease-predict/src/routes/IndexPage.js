import React from 'react';
import { connect } from 'dva';
import styles from './IndexPage.css';
import { Button } from 'antd';
import { Link } from 'dva/router';
import { PieChartOutlined, BranchesOutlined } from '@ant-design/icons';
import 'antd/dist/antd.css';

function IndexPage() {
  return (
    <div className={styles.normal}>
      <h1 className={styles.title}>儿童遗传病面部识别demo</h1>
      <Link to="/predict_nn" className={styles.chooseButton}>
        <Button type="primary" shape="round" icon={<PieChartOutlined />} size="large">
          深度学习预测
        </Button>
      </Link>
      <Link to="/predict" className={styles.chooseButton}>
        <Button type="primary" shape="round" icon={<PieChartOutlined />} size="large">
          传统机器学习预测
        </Button>
      </Link>
      <Link to='/upload/train' className={styles.chooseButton}>
        <Button type="primary" shape="round" icon={<BranchesOutlined />} size="large">
          训练
        </Button>
      </Link>
      <ul className={styles.list}>
        <li>点击深度学习预测按钮使用预训练深度学习模型预估患病风险</li>
        <li>点击传统机器学习预测按钮使用传统机器学习模型预估患病风险</li>
        <li>点击训练按钮上传图片压缩包自动训练模型</li>
      </ul>
    </div>
  );
}

IndexPage.propTypes = {
};

export default connect()(IndexPage);
//https://stackoverflow.com/questions/51455898/how-to-package-an-electron-app-and-flask-server-into-one-executable
//https://blog.yasking.org/a/electron-with-dvajs.html