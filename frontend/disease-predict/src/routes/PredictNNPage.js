import { useState, useEffect } from 'react';
import { connect } from 'dva';
import axios from 'axios';
import globalStyles from './IndexPage.css';
import LocalStyles from './PredictPage.css';
import { Button } from 'antd';
import { Link } from 'dva/router';
import ReturnIndex from '../components/ReturnIndex';

function IndexPage() {
  const [modelList, setModelList] = useState([])

  useEffect(() => {
    axios.get('http://localhost:5000/nn_model_list').then((res) => {
      setModelList(res.data.model_list)
    })
  })

  const list_models = (model_list) => {
    let res = []
    for (let index in model_list) {
      res.push(
        <div className={LocalStyles.diseaseButton} key={index}>
          <Link to={{
                      pathname:"/upload/predict_nn",
                      state: {
                        disease: model_list[index][0]
                      }
                    }}>
            <Button type="primary" shape="round" size="large">
              {model_list[index][0]}
            </Button>
            <div>训练准确率：{(model_list[index][1]*100).toFixed(2)}%</div>
          </Link>
        </div>
      )
    }
    return res
  }

  return (
    <div className={globalStyles.normal}>
      <h1 className={globalStyles.title}>预训练模型预估</h1>
      <div className={LocalStyles.diseaseButtonDiv}>
          {list_models(modelList)}
      </div>
      <ul className={globalStyles.list}>
        <li>上述列表包括预训练模型与用户训练模型</li>
        <li>点击对应按钮进入相应预测页面</li>
        <li>点击mutil-disease按钮进入多疾病预测页面</li>
      </ul>
      <ReturnIndex />
    </div>
  );
}

IndexPage.propTypes = {
};

export default connect()(IndexPage);
