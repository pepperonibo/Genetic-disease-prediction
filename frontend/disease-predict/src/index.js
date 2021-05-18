import dva from 'dva';
import './index.css';
// const proxy = require('http-proxy-middleware');

// 1. Initialize
const app = dva();

// 2. Plugins
// app.use(
//     '/api',
//     proxy(
//         target="http://localhost:5000",
//         changeOrigin= true,
//     )
// );

// 3. Model
// app.model(require('./models/example').default);

// 4. Router
app.router(require('./router').default);

// 5. Start
app.start('#root');
