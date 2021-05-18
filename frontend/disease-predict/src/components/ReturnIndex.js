import { RollbackOutlined } from '@ant-design/icons';
import { Button } from 'antd';
import { Link } from 'dva/router';

function ReturnIndex() {
    return (
        <Link to="/">
            <Button type="primary" shape="round" icon={<RollbackOutlined />} size="large">
            返回首页
            </Button>
        </Link>
    )
}

export default ReturnIndex;