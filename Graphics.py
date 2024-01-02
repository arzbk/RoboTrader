from lightweight_charts import Chart
import time

class StockChart(Chart):

    def __init__(self, **kwargs):

        # Get custom kwarg vals before passing to parent
        self.include_table = kwargs.pop('include_table')
        if self.include_table:
            self.table_cols = kwargs.pop('table_cols')

        # Instantiate Chart class
        self.parent = super().__init__(**kwargs)

        # Create table and format columns
        if self.include_table:
            self.table = self.create_table(
                width=0.95,
                height=0.40,
                headings=self.table_cols,
                widths=tuple([1.00 / float(len(self.table_cols))]*len(self.table_cols)),
                alignments=tuple(['left']*len(self.table_cols)),
                position='right',
                draggable=True,
            )
            """
            for col in self.table_cols:
                self.table.format(col, f'$ {self.table.VALUE}')
            """

        self.ui_metrics = None

    def show(self, block=False):
        super().show(block)
        self.table.visible(True)

    def reset(self, lead_data):

        # If first iteration has completed already...
        if self.ui_metrics:

            for i in range(1, 4):
                watermark_text = (i * " ") + "Trial Complete. Resetting" + (i * ".")
                self.watermark(watermark_text)
                time.sleep(0.2)

            self.watermark("")
            self.clear_markers()

        # Set leading data
        self.set(lead_data)

    # Update metrics for the chart table
    def update_metrics(self, values):

        if self.ui_metrics:
            for i, metric in enumerate(self.table_cols):
                self.ui_metrics[metric] = values[i]
        else:
            self.ui_metrics = self.table.new_row(
                values
            )

    # Marks actions taken on the appropriate timestep of the chart
    def mark_action(self, action_type, sub_type=None):

        if action_type == "BUY":
            self.marker(shape="circle", color="#FFFF00")

        elif action_type == "SELL" and sub_type == "PROFIT":
            self.marker(shape="arrow_up", color="#008000")

        elif action_type == "SELL" and sub_type == "LOSS":
            self.marker(shape="arrow_down", color="#FF0000")


    # Update chart with next step
    def add_step_data(self, step_data):
        self.update(step_data)