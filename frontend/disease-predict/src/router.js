import React from 'react';
import { Router, Route, Switch } from 'dva/router';
import IndexPage from './routes/IndexPage';
import PredictPage from './routes/PredictPage'
import PredictNNPage from './routes/PredictNNPage'
import UploadPredictImagePage from './routes/UploadPredictImagePage'
import UploadNNPredictImagePage from './routes/UploadNNPredictImagePage'
import TrainPage from './routes/TrainPage'

function RouterConfig({ history }) {
  return (
    <Router history={history}>
      <Switch>
        <Route path="/" exact component={IndexPage} />
        <Route path="/predict" exact component={PredictPage} />
        <Route path="/predict_nn" exact component={PredictNNPage} />
        <Route path="/upload/predict" exact component={UploadPredictImagePage} />
        <Route path="/upload/predict_nn" exact component={UploadNNPredictImagePage} />
        <Route path="/upload/train" exact component={TrainPage} />
      </Switch>
    </Router>
  );
}

export default RouterConfig;
