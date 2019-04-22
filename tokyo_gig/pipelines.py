# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy.exporters import CsvItemExporter


class WriteItemPipeline(object):

    def __init__(self):
        self.filename = 'tokyo_gig_guide_past_events_mid_219.csv' #change filename to reflect current page batch 

    def open_spider(self, spider):
        self.csvfile = open(self.filename, 'wb')
        self.exporter = CsvItemExporter(self.csvfile)
        #set column order in csv:  
        self.exporter.fields_to_export = ['title', 'date', 'category', 'venue', 'venue_url', 'address', 'access_url', 'area', 'closest_station', 'advance_price', 'door_price']
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.csvfile.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item
    
    