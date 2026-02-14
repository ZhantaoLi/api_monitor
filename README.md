# API Monitor

一个独立于 `api_test/` 的 API 可用性巡检项目，目标是：

1. 统一管理多个 API 目标（`base_url + api_key`）。
2. 定时执行模型测活（默认每个目标 30 分钟）。
3. 提供主界面查看健康状态、成功率、最近执行结果。
4. 提供详细日志可视化页面（图表 + 可排序表格 + 详情面板）。

> 当前设计按你的要求：`api_key` 直接保存在本地 SQLite（不走环境变量）。

## 1. 功能概览

- API 目标管理
  - 新增目标、启用/禁用目标、手动立即执行目标。
- 自动调度
  - 后台每分钟扫描一次“到期目标”，到期后触发检测。
- 结果存储
  - 摘要数据写入 SQLite。
  - 明细数据写入 JSONL（每次运行一个文件）。
- Web 可视化
  - 主界面：目标状态、成功率、最近运行时间、快捷操作。
  - 日志页面：筛选、排序、图表交互、单条日志详情。

## 2. 技术栈

- 后端：FastAPI
- 调度：APScheduler
- 存储：SQLite
- 前端：原生 HTML + CSS + JS + Chart.js
- 部署：Docker / Docker Compose（也可直接 `uvicorn`）

## 3. 目录结构（重点）

```text
api_monitor/
├─ app.py                  # FastAPI 入口，路由定义，静态页面挂载
├─ monitor.py              # 调度器与测活执行逻辑（请求、路由判断、写日志）
├─ db.py                   # SQLite 数据层（建表、增删改查、运行结果落库）
├─ requirements.txt        # Python 依赖
├─ Dockerfile              # 容器构建定义
├─ docker-compose.yml      # 容器启动编排
├─ .dockerignore           # 构建忽略规则
├─ README.md               # 当前文档
├─ web/
│  ├─ index.html           # 主界面（目标管理 + 状态总览）
│  └─ log_viewer.html      # 日志可视化页面（按 target_id 查看）
└─ data/
   ├─ registry.db          # SQLite 数据库（运行后自动生成）
   └─ logs/
      └─ target_<id>_<ts>.jsonl   # 每次检测的明细日志
```

说明：

- `.venv/`、`__pycache__/` 是本地运行产物，不属于核心代码结构。
- `data/` 建议持久化（Docker 里已通过 volume 挂载）。

## 4. 核心流程

1. 用户在主界面新增目标。
2. `app.py` 调用 `db.py` 写入 `targets`。
3. `monitor.py` 调度器扫描到期目标并执行检测。
4. 每个模型检测结果：
   - 写入 `data/logs/*.jsonl`（明细）
   - 写入 `run_models`（结构化明细）
5. 单次运行摘要写入 `runs`，并回写 `targets.last_*` 字段。
6. 主界面和日志页通过 `/api/*` 接口读取数据并渲染。

## 5. 主要接口（上手够用）

- `GET /api/health`
  - 服务健康检查。
- `GET /api/dashboard`
  - 主界面统计卡片数据。
- `GET /api/targets`
  - 目标列表。
- `POST /api/targets`
  - 新增目标。
- `PATCH /api/targets/{target_id}`
  - 更新目标（如启用/禁用）。
- `POST /api/targets/{target_id}/run`
  - 立即触发该目标检测。
- `GET /api/targets/{target_id}/runs`
  - 查看该目标的运行历史摘要。
- `GET /api/targets/{target_id}/logs?scope=latest|all&limit=...`
  - 获取日志可视化所需明细。

## 6. 本地运行（不走 Docker）

```bash
cd api_monitor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

访问：

- 主界面：`http://127.0.0.1:8000/`
- 日志页：`http://127.0.0.1:8000/viewer.html?target_id=1`

## 7. Docker 运行

```bash
cd api_monitor
docker compose up -d --build
docker compose ps
```

访问：

- `http://<你的IP>:8000/`

停止：

```bash
docker compose down
```

## 8. 数据表说明（简版）

- `targets`
  - 目标配置与最近状态（`last_status`、`last_success`、`last_fail` 等）。
- `runs`
  - 每次运行摘要（开始/结束时间、成功失败数量、错误信息）。
- `run_models`
  - 每个模型的检测明细（protocol、model、duration、success、error 等）。

## 9. 常见操作

1. 新增一个目标
   - 打开主界面，填写名称、URL、Key，保存。
2. 立即测试目标
   - 在目标行点击“立即运行”。
3. 查看目标日志
   - 在目标行点击“查看日志”。
4. 只看最近一次数据
   - 在日志页 `Scope` 选 `latest`。
5. 分析失败原因
   - 日志页筛选 `仅失败`，点击表格行看右侧 `Error` 详情。

## 10. 注意事项

- `api_key` 明文存储在 `registry.db`，请自行控制主机访问权限。
- 若模型数量非常大，可在目标配置中设置 `max_models` 限制单次巡检量。
- 如果部署在公网，建议加 Nginx 反代、IP 白名单或鉴权层（当前默认无登录）。

---

如果后续要做二期扩展，建议优先做：

1. 目标分组与标签（按渠道/客户分组）。
2. 告警通知（失败阈值触发 Telegram/飞书/Webhook）。
3. 日志归档策略（按天分表或自动清理）。
