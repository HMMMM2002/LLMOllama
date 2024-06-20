from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ollama
from langchain_community.document_loaders import PyPDFLoader

class loadInfo:
    def loadDocument(inputDocument):
        loader = 
        
        #我想在这里根据传进来的文件名后缀判断它需要用那种loader，你觉得需要放在这吗？我想一想，我现在就是在构思，
        # 因为我找到的那个tutorial
        #他用的那个 MongoDBAtlasVectorSearch这个，感觉就和 aws bedrock差不多，要收费，就是你只管用就好了，什么都给你写好了
        #原来是这样，这东西在langchain库里但是也还要收费啊，因为不是ollama，langchain相当于是emmm提供一个链式结构？？
        # aws也得用langchain好吧我也看到他那个收费标准什么的了hh，
        #loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)我看到有这么个用法，
        #应该是用来读取某个文件夹下所有文件的，我发截图给你了，glob好像是用来指定只加载后缀为什么的文件，我觉得一开始不用想的
        #很复杂，觉得会有很多种文件类型，我们就先假定是pdf或者txt好了那就pdf好了，因为之后再加这种判断也是OK的，应该也是简单的