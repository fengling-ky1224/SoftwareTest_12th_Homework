<template>
<div>
	<div :style='{"padding":"0px 7%","margin":"10px auto","background":"#5bb450","display":"flex","width":"100%","justifyContent":"space-between","height":"40px"}' class="breadcrumb-preview">
		<el-breadcrumb :separator="'Ξ'" :style='{"fontSize":"14px","lineHeight":"40px"}'>
			<el-breadcrumb-item>首页</el-breadcrumb-item>
			<el-breadcrumb-item v-for="(item, index) in breadcrumbItem" :key="index">{{item.name}}</el-breadcrumb-item>
		</el-breadcrumb>
	</div>
	<div class="breadcrumb-preview">
        <el-form :inline="true" :model="formSearch" class="list-form-pv" style="justify-content: center;display: flex;">
            <div class="section-btn">
                <el-form-item>
                  <el-input v-model="title" placeholder="标题"></el-input>
                </el-form-item>
              <el-button type="primary" @click="getNewsList(1)"><i class="el-icon-search"></i>查询</el-button>
            </div>
        </el-form>
	</div>
	
	<div class="news-preview" :style='{"width":"100%","padding":"10px 7% 0px","margin":"0px auto","position":"relative","background":"none"}'>
				
				
				
				
				
				
				<!-- 样式七 -->
		<div class="list7 index-pv1" :style='{"width":"100%","flexWrap":"wrap","background":"none","justifyContent":"space-between","display":"flex","height":"auto"}'>
			<div :style='{"border":"1px solid #bdefb6","padding":"10px","boxShadow":"inset 0px 0px 56px 0px #c5f1c0","margin":"0 0 20px","background":"#fff","width":"48%","position":"relative","height":"auto"}' v-for="item in newsList" :key="item.id" @click="toNewsDetail(item)" class="list-item animation-box">
				<div :style='{"padding":"0 10px","borderColor":"#b5e8af","whiteSpace":"nowrap","overflow":"hidden","color":"#333","background":"none","borderWidth":"0 0 1px","lineHeight":"32px","fontSize":"14px","textOverflow":"ellipsis","borderStyle":"dashed","fontWeight":"600"}' class="title ">{{item.title}}</div>
				<div :style='{"margin":"4px 0 0","overflow":"hidden","color":"#666","lineHeight":"24px","fontSize":"14px","textIndent":"2em","height":"72px"}' class="introduction ">{{item.introduction}}</div>
				<div :style='{"lineHeight":"32px","fontSize":"12px","position":"absolute","right":"20px","color":"#555","top":"10px"}' class="time ">{{item.addtime}}</div>
				<div :style='{"padding":"0 10px","fontSize":"14px","lineHeight":"32px","color":"#fff","background":"#333","display":"none"}' class="tags">新闻动态</div>
			</div>
		</div>
				
		<el-pagination
		  background
		  class="pagination"
		  :pager-count="7"
		  :page-size="pageSize"
		  :page-sizes="pageSizes"
		  prev-text="<"
		  next-text=">"
		  :hide-on-single-page="true"
		  :layout='["total","prev","pager","next","sizes","jumper"].join()'
		  :total="total"
		  :style='{"padding":"20px 7% 20px","margin":"0","whiteSpace":"nowrap","color":"#333","textAlign":"center","background":"#f3f3f3","width":"100%","fontWeight":"500"}'
		  @current-change="curChange"
		  @prev-click="prevClick"
		  @next-click="nextClick"
		></el-pagination>
	</div>
</div>
</template>

<script>
  export default {
    //数据集合
    data() {
      return {
        baseUrl: this.$config.baseUrl,
        breadcrumbItem: [
          {
            name: '通知公告'
          }
        ],
        newsList: [],
        total: 1,
        pageSize: 10,pageSizes: [10,20,30,50],
        totalPage: 1,
		layouts: '',
        title: '',
      }
    },
    created() {
		this.getNewsList(1);
    },
    //方法集合
    methods: {
      getNewsList(page) {
        let params = {page, limit: this.pageSize,sort:'addtime',order:'desc'};
        let searchWhere = {};
        if(this.title != '') searchWhere.title = '%' + this.title + '%';
        this.$http.get('news/list', {params: Object.assign(params, searchWhere)}).then(res => {
          if (res.data.code == 0) {
            this.newsList = res.data.data.list;
            this.total = res.data.data.total;
            this.pageSize = res.data.data.pageSize;this.pageSizes = [this.pageSize, this.pageSize*2, this.pageSize*3, this.pageSize*5];
            this.totalPage = res.data.data.totalPage;
          }
        });
      },
      curChange(page) {
        this.getNewsList(page);
      },
      prevClick(page) {
        this.getNewsList(page);
      },
      nextClick(page) {
        this.getNewsList(page);
      },
      toNewsDetail(item) {
        this.$router.push({path: '/index/newsDetail', query: {detailObj: JSON.stringify(item)}});
      }
    }
  }
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
	.news-wrap {
	  width: 1000px;
	  margin: 0 auto;
	}
	.news-item {
	  height: 200px;
	  width: auto;
	  margin-bottom: 30px;
	  cursor: pointer;
	}
	.news-item-text {
	  width: calc(100% - 215px);
	  display: inline-block;
	  vertical-align: top;
	  margin-left: 15px;
	}
	.news-item-text-title {
	  font-size: 16px;
	  margin-bottom: 15px;
	}
	.news-item-text-intro {
	  font-size: 14px;
	  color: #666;
	}
	
	.hide1 {
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 1;
		-webkit-box-orient: vertical;
		overflow:hidden;
		text-overflow:ellipsis;
	}
	.hide2 {
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow:hidden;
		text-overflow:ellipsis;
	}
	.hide3 {
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow:hidden;
		text-overflow:ellipsis;
	}
	
	.index-pv1 .animation-box {
		transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
		z-index: initial;
	}
	
	.index-pv1 .animation-box:hover {
				transform: translate3d(0px, -10px, 0px);
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
				z-index: 1;
	}
	
	.index-pv1 .animation-box img {
		transform: rotate(0deg) scale(1) skew(0deg, 0deg) translate3d(0px, 0px, 0px);
	}
	
	.index-pv1 .animation-box img:hover {
				-webkit-perspective: 1000px;
				perspective: 1000px;
				transition: 0.3s;
			}
		
	.el-pagination /deep/ .el-pagination__total {
				margin: 0 10px 0 0;
				color: #666;
				font-weight: 400;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.el-pagination /deep/ .btn-prev {
				border: none;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #666;
				background: #ffffff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				min-width: 35px;
				height: 28px;
			}
	
	.el-pagination /deep/ .btn-next {
				border: none;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #666;
				background: #ffffff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				min-width: 35px;
				height: 28px;
			}
	
	.el-pagination /deep/ .btn-prev:disabled {
				border: none;
				cursor: not-allowed;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #C0C4CC;
				background: #ffffff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.el-pagination /deep/ .btn-next:disabled {
				border: none;
				cursor: not-allowed;
				border-radius: 2px;
				padding: 0;
				margin: 0 5px;
				color: #C0C4CC;
				background: #ffffff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pager {
				padding: 0;
				margin: 0;
				display: inline-block;
				vertical-align: top;
			}
	
	.el-pagination /deep/ .el-pager .number {
				cursor: pointer;
				padding: 0 4px;
				margin: 0 5px;
				color: #666;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				border-radius: 2px;
				background: #ffffff;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pager .number:hover {
				cursor: pointer;
				border-radius: 2px;
				padding: 0 4px;
				margin: 0 5px;
				background: #ffffff;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pager .number.active {
				cursor: default;
				padding: 0 4px;
				margin: 0 5px;
				color: #FFF;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				border-radius: 2px;
				background: #59B450;
				text-align: center;
				min-width: 30px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pagination__sizes {
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pagination__sizes .el-input {
				margin: 0 5px;
				width: 100px;
				position: relative;
			}
	
	.el-pagination /deep/ .el-pagination__sizes .el-input .el-input__inner {
				border: 1px solid #DCDFE6;
				cursor: pointer;
				padding: 0 25px 0 8px;
				color: #606266;
				display: inline-block;
				font-size: 13px;
				line-height: 28px;
				border-radius: 3px;
				outline: 0;
				background: #FFF;
				width: 100%;
				text-align: center;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pagination__sizes .el-input span.el-input__suffix {
				top: 0;
				position: absolute;
				right: 0;
				height: 100%;
			}
	
	.el-pagination /deep/ .el-pagination__sizes .el-input .el-input__suffix .el-select__caret {
				cursor: pointer;
				color: #C0C4CC;
				width: 25px;
				font-size: 14px;
				line-height: 28px;
				text-align: center;
			}
	
	.el-pagination /deep/ .el-pagination__jump {
				margin: 0 0 0 24px;
				color: #606266;
				display: inline-block;
				vertical-align: top;
				font-size: 13px;
				line-height: 28px;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pagination__jump .el-input {
				border-radius: 3px;
				padding: 0 2px;
				margin: 0 2px;
				display: inline-block;
				width: 50px;
				font-size: 14px;
				line-height: 18px;
				position: relative;
				text-align: center;
				height: 28px;
			}
	
	.el-pagination /deep/ .el-pagination__jump .el-input .el-input__inner {
				border: 1px solid #DCDFE6;
				cursor: pointer;
				padding: 0 3px;
				color: #606266;
				display: inline-block;
				font-size: 14px;
				line-height: 28px;
				border-radius: 3px;
				outline: 0;
				background: #FFF;
				width: 100%;
				text-align: center;
				height: 28px;
			}
</style>
