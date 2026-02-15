# API Monitor

`API Monitor` 是一个用于批量管理 API 渠道并周期性执行模型测活的小项目。  

## 项目能力

- 管理多个目标渠道（`name + base_url + api_key`）
- 定时巡检（每分钟扫描一次到期目标，默认目标间隔 30 分钟）
- 模型并发检测（目标内并发默认 3，目标间并发默认 2）
- 检测结果双写
  - 结构化数据写入 SQLite
  - 明细日志写入 JSONL（每次运行一个文件）
-   可视化页面
    -   支持深色/浅色模式切换
    -   仪表盘：渠道状态、模型健康、手动触发、删除渠道
    -   日志查看：筛选、详情面板
    -   分析页：图表统计与错误分布

## 技术栈

- 后端：FastAPI + Pydantic
- 调度：APScheduler
- 存储：SQLite（WAL）
- 前端：HTML + Tailwind + Alpine.js + Chart.js
- 部署：Docker Compose / `uvicorn`

## 目录结构

```text
api_monitor/
├─ app.py                    # FastAPI 入口与 API 路由
├─ monitor.py                # 调度与测活执行
├─ db.py                     # SQLite 数据访问层
├─ requirements.txt          # Python 依赖
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ .gitignore
├─ .gitattributes
├─ README.md
├─ web/
│  ├─ index.html             # 主界面
│  ├─ log_viewer.html        # 日志详情页面
│  ├─ analysis.html          # 统计分析页面
│  └─ assets/
│     ├─ main.js
│     └─ styles.css
└─ data/                     # 运行后生成（默认被 .gitignore 忽略）
   ├─ registry.db
   └─ logs/
      └─ target_<id>_<yyyyMMdd_HHmmss>.jsonl
```

## Linux Docker 部署

### 1. 拉取代码并进入项目

```bash
git clone https://github.com/ZhantaoLi/api_monitor
cd ./api_monitor
```

### 2. 启动服务

```bash
docker compose up -d --build
```

### 3. 验证服务

```bash
curl http://127.0.0.1:8081/api/health
```

访问页面：

- 主界面：`http://<服务器IP>:8081/`
- 日志页：`http://<服务器IP>:8081/viewer.html?target_id=1`
- 分析页：`http://<服务器IP>:8081/analysis.html?target_id=1`

### 4. 常用项目运维命令

```bash
# 查看日志
docker compose logs -f

# 重启
docker compose restart

# 更新后重建
git pull
docker compose up -d --build

# 停止并删除容器（保留 data/ 数据）
docker compose down
```

## 本地运行（不使用 Docker）

```bash
cd api_monitor
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8081
```

## 核心流程

1. 新增目标后写入 `targets` 表。
2. 主界面新增成功后，会立刻调用一次 `POST /api/targets/{id}/run` 触发即时检测。
3. 调度器每分钟调用 `scan_due_targets`，筛选到期目标。
4. 对目标执行：
   - `GET /v1/models` 拉取模型列表
   - 按规则分配到 `chat/responses/anthropic/gemini` 路由
   - 并发检测并记录每个模型结果
5. 结果入库：
   - `runs`：单次运行摘要
   - `run_models`：每个模型明细
   - `targets.last_*`：目标最近状态快照
6. 同时落盘 JSONL，便于外部排查与归档。

## 后端接口（主要）

- `GET /api/health`：服务健康与正在运行的目标
- `GET /api/dashboard`：仪表盘统计
- `GET /api/targets`：目标列表
- `GET /api/targets/{target_id}`：单目标详情
- `POST /api/targets`：新增目标
- `PATCH /api/targets/{target_id}`：更新目标（启用/禁用、参数等）
- `POST /api/targets/{target_id}/run`：立即运行目标
- `DELETE /api/targets/{target_id}`：删除目标
- `GET /api/targets/{target_id}/runs`：目标运行历史
- `GET /api/targets/{target_id}/logs`：日志数据（支持 `scope=latest|all`）

## 目标配置字段（默认值）

- `interval_min`: `30`
- `timeout_s`: `30.0`
- `verify_ssl`: `false`
- `max_models`: `0`（0 表示不限制）
- `anthropic_version`: `2025-09-29`
- `source_url`: `null`
- `prompt`:  
  `What is the exact model identifier (model string) you are using for this chat/session?`

## 数据说明

- 数据库：`data/registry.db`
- 日志文件：`data/logs/target_<id>_<timestamp>.jsonl`
- JSONL 记录核心字段：
  - `protocol`, `model`, `success`, `duration`, `status_code`
  - `error`, `content`, `route`, `endpoint`, `timestamp`

## data 目录清理策略

- 清理对象只包含 `data/logs/*.jsonl`，不会删除 `data/registry.db`。
- 仅按总大小清理：当 `data/logs` 总体积超过阈值时，自动从最旧日志开始删除。
- 配置项：
  - `LOG_CLEANUP_ENABLED=1`
  - `LOG_MAX_SIZE_MB=512`（总大小上限，单位 MB）

## 注意事项

- 当前默认无登录鉴权；公网部署请自行加反向代理和访问控制。
- `api_key` 明文存储在 SQLite。

## 致谢
  
 - https://github.com/chxcodepro/model-check
