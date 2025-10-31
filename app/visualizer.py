"""
图表处理模块
"""
import io
import base64
import json
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib
from app.models import ChartOutput

matplotlib.use('Agg')


class ChartCapture:
    """图表捕获器"""

    def __init__(self):
        self.charts: List[ChartOutput] = []
        self.matplotlib_figures = []
        self.plotly_figures = []

    def capture_matplotlib(self) -> List[ChartOutput]:
        """
        捕获所有 matplotlib 图表

        Returns:
            图表列表
        """
        charts = []
        figures = plt.get_fignums()

        for fig_num in figures:
            try:
                fig = plt.figure(fig_num)

                # 转换为 PNG base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()

                # 获取图表尺寸
                width, height = fig.get_size_inches() * fig.dpi

                chart = ChartOutput(
                    type="matplotlib",
                    format="png",
                    data=img_base64,
                    width=int(width),
                    height=int(height)
                )
                charts.append(chart)

            except Exception as e:
                print(f"捕获 matplotlib 图表失败: {e}")
            finally:
                plt.close(fig)

        return charts

    def capture_plotly(self, local_vars: dict) -> List[ChartOutput]:
        """
        捕获 plotly 图表

        Args:
            local_vars: 执行后的局部变量字典

        Returns:
            图表列表
        """
        charts = []

        # 查找所有 plotly 图表对象
        for var_name, var_value in local_vars.items():
            try:
                # 检查是否是 plotly 图表
                if hasattr(var_value, 'to_json') and hasattr(var_value, 'data'):
                    # 是 plotly.graph_objects.Figure 对象
                    fig_json = var_value.to_json()

                    chart = ChartOutput(
                        type="plotly",
                        format="json",
                        data=fig_json,
                        width=var_value.layout.width if var_value.layout.width else 700,
                        height=var_value.layout.height if var_value.layout.height else 450
                    )
                    charts.append(chart)

            except Exception as e:
                print(f"捕获 plotly 图表 '{var_name}' 失败: {e}")

        return charts

    def capture_all(self, local_vars: dict) -> List[ChartOutput]:
        """
        捕获所有图表

        Args:
            local_vars: 执行后的局部变量字典

        Returns:
            所有图表列表
        """
        all_charts = []

        # 捕获 matplotlib 图表
        matplotlib_charts = self.capture_matplotlib()
        all_charts.extend(matplotlib_charts)

        # 捕获 plotly 图表
        plotly_charts = self.capture_plotly(local_vars)
        all_charts.extend(plotly_charts)

        return all_charts

    @staticmethod
    def clear_all():
        """清除所有图表"""
        plt.close('all')


class DataFrameCapture:
    """DataFrame 捕获器"""

    @staticmethod
    def capture_dataframes(local_vars: dict) -> list:
        """
        捕获所有 pandas DataFrame

        Args:
            local_vars: 执行后的局部变量字典

        Returns:
            DataFrame 列表
        """
        import pandas as pd

        dataframes = []

        for var_name, var_value in local_vars.items():
            try:
                if isinstance(var_value, pd.DataFrame):
                    # 限制显示行数
                    max_rows = 100
                    df_display = var_value.head(max_rows) if len(var_value) > max_rows else var_value

                    # 转换为 HTML
                    html = df_display.to_html(
                        index=True,
                        classes='table table-striped table-bordered',
                        border=0,
                        max_rows=max_rows
                    )

                    dataframe_info = {
                        "name": var_name,
                        "html": html,
                        "shape": var_value.shape,
                        "columns": var_value.columns.tolist()
                    }
                    dataframes.append(dataframe_info)

            except Exception as e:
                print(f"捕获 DataFrame '{var_name}' 失败: {e}")

        return dataframes


def format_chart_for_frontend(chart: ChartOutput) -> dict:
    """
    格式化图表数据供前端使用

    Args:
        chart: 图表对象

    Returns:
        前端友好的字典格式
    """
    if chart.type == "matplotlib":
        return {
            "type": "image",
            "format": "png",
            "src": f"data:image/png;base64,{chart.data}",
            "width": chart.width,
            "height": chart.height
        }
    elif chart.type == "plotly":
        return {
            "type": "plotly",
            "format": "json",
            "data": json.loads(chart.data),
            "width": chart.width,
            "height": chart.height
        }
    else:
        return {}
