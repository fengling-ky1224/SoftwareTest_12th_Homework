<template>
<div>
	<div class="container" :style='{"minHeight":"100vh","alignItems":"center","background":"url(http://codegen.caihongy.cn/20210504/6f5e101948464749875bc28f19675784.jpg) no-repeat center top / 100% 100%","display":"flex","width":"100%","backgroundSize":"cover","backgroundPosition":"center center","backgroundRepeat":"no-repeat","justifyContent":"center"}'>
		<el-form ref="loginForm" :model="loginForm" :style='{"border":"1px solid #bdefb6","padding":"30px 40px","boxShadow":"0 0px 0px rgba(0, 0, 0, .1)","margin":"0","borderRadius":"30px","flexWrap":"wrap","background":"rgba(255,255,255,.9)","display":"flex","width":"880px","position":"relative","justifyContent":"space-between","height":"auto"}' :rules="rules">
			<div v-if="false" :style='{"margin":"0 0 10px 0","color":"rgba(64, 158, 255, 1)","textAlign":"center","width":"100%","lineHeight":"44px","fontSize":"20px","textShadow":"4px 4px 2px rgba(64, 158, 255, .5)"}'>USER / LOGIN</div>
			<div v-if="true" :style='{"boxShadow":"inset 0px 0px 32px 0px #c5f1c0","margin":"0 0 20px","borderColor":"#bdefb6","color":"#5bb450","textAlign":"center","top":"-80px","left":"10%","borderWidth":"1px 4px","background":"#fff","width":"80%","lineHeight":"44px","fontSize":"24px","position":"absolute","borderStyle":"solid","textShadow":"0px 0px 0px rgba(64, 158, 255, .5)"}'>校园便利平台登录</div>
			<el-form-item v-if="loginType==1" class="list-item" :style='{"width":"67%","padding":"0 0 0 100px","margin":"0 0 24px","position":"relative","flexWrap":"wrap","display":"flex"}' prop="username">
				<div v-if="true" :style='{"color":"#333","left":"-100px","textAlign":"right","width":"100px","lineHeight":"44px","fontSize":"14px","position":"absolute"}'>账号：</div>
				<input :style='{"border":"1px solid #bdefb6","padding":"0 10px","boxShadow":"inset 0px 0px 12px 0px #c5f1c0","outline":"0px solid #efefef","color":"#333","outlineOffset":"4px","borderRadius":"8px","flex":"1","width":"100%","fontSize":"14px","height":"40px"}' v-model="loginForm.username" placeholder="请输入账户">
			</el-form-item>
			<el-form-item v-if="loginType==1" class="list-item" :style='{"width":"67%","padding":"0 0 0 100px","margin":"0 0 24px","position":"relative","flexWrap":"wrap","display":"flex"}' prop="password">
				<div v-if="true" :style='{"color":"#333","left":"-100px","textAlign":"right","width":"100px","lineHeight":"44px","fontSize":"14px","position":"absolute"}'>密码：</div>
				<input :style='{"border":"1px solid #bdefb6","padding":"0 10px","boxShadow":"inset 0px 0px 12px 0px #c5f1c0","outline":"0px solid #efefef","color":"#333","outlineOffset":"4px","borderRadius":"8px","flex":"1","width":"100%","fontSize":"14px","height":"40px"}' v-model="loginForm.password" placeholder="请输入密码" type="password">
			</el-form-item>
			<el-form-item v-if="roles.length>1" class="list-type" :style='{"width":"67%","padding":"0 0 0 100px","margin":"20px 0 10px"}' prop="role">
				<el-radio v-model="loginForm.tableName" :label="item.tableName" v-for="(item, index) in roles" :key="index" @change.native="getCurrentRow(item)">{{item.roleName}}</el-radio>
			</el-form-item>
			<el-form-item :style='{"border":"1px solid #bdefb6","boxShadow":"inset 0px 0px 56px 0px #c5f1c0","margin":"0 40px 0px 0","borderRadius":"20%","background":"#fff","width":"25%","position":"absolute","right":"0","height":"calc(100% - 60px)"}'>
				<el-button v-if="loginType==1" :style='{"border":"0","cursor":"pointer","padding":"0 10px","margin":"30px 23% 0","borderColor":"#59B450","color":"#333","minWidth":"51%","outline":"none","borderRadius":"4px","background":"#c6ecc6","borderWidth":"4px 0","width":"auto","fontSize":"14px","borderStyle":"solid","height":"44px"}' @click="submitForm('loginForm')">登录</el-button>
				<el-button v-if="loginType==1" :style='{"border":"0","cursor":"pointer","padding":"0 24px","margin":"0 5px","outline":"none","color":"#fff","borderRadius":"4px","background":"rgba(64, 158, 255, 1)","display":"none","width":"auto","fontSize":"14px","height":"44px"}' @click="resetForm('loginForm')">重置</el-button>
			</el-form-item>
			<div :style='{"margin":"0 40px 0 0","alignItems":"center","top":"114px","flexWrap":"wrap","display":"flex","width":"25%","position":"absolute","right":"4px","justifyContent":"center","height":"auto"}'>
			<router-link :style='{"cursor":"pointer","padding":"3px 10px","margin":"5px 20%","borderColor":"#999","color":"#333","textAlign":"center","background":"#eee","borderWidth":"4px 0","width":"50%","fontSize":"14px","textDecoration":"none","borderStyle":"solid"}' :to="{path: '/register', query: {role: item.tableName,pageFlag:'register'}}" v-if="item.hasFrontRegister=='是'" v-for="(item, index) in roles" :key="index">注册{{item.roleName.replace('注册','')}}</router-link>
			</div>
		</el-form>
    </div>
</div>
</template>

<script>

export default {
	//数据集合
	data() {
		return {
            baseUrl: this.$config.baseUrl,
            loginType: 1,
			roleMenus: [{"backMenu":[{"child":[{"appFrontIcon":"cuIcon-form","buttons":["新增","查看","修改","删除"],"menu":"用户","menuJump":"列表","tableName":"yonghu"}],"menu":"用户管理"},{"child":[{"appFrontIcon":"cuIcon-shop","buttons":["新增","查看","修改","删除"],"menu":"商品类型","menuJump":"列表","tableName":"shangpinleixing"}],"menu":"商品类型管理"},{"child":[{"appFrontIcon":"cuIcon-time","buttons":["新增","查看","修改","删除","查看评论"],"menu":"二手商品","menuJump":"列表","tableName":"ershoushangpin"}],"menu":"二手商品管理"},{"child":[{"appFrontIcon":"cuIcon-flashlightopen","buttons":["新增","查看","修改","删除","领单"],"menu":"校园跑腿","menuJump":"列表","tableName":"xiaoyuanpaotui"}],"menu":"校园跑腿管理"},{"child":[{"appFrontIcon":"cuIcon-form","buttons":["查看","修改","删除","完成登记"],"menu":"跑腿领单","menuJump":"列表","tableName":"paotuilingdan"}],"menu":"跑腿领单管理"},{"child":[{"appFrontIcon":"cuIcon-time","buttons":["查看","修改","删除","评价"],"menu":"完成订单","menuJump":"列表","tableName":"wanchengdingdan"}],"menu":"完成订单管理"},{"child":[{"appFrontIcon":"cuIcon-goodsnew","buttons":["查看","修改","删除"],"menu":"评价信息","menuJump":"列表","tableName":"pingjiaxinxi"}],"menu":"评价信息管理"},{"child":[{"appFrontIcon":"cuIcon-similar","buttons":["新增","查看","修改","删除"],"menu":"轮播图管理","tableName":"config"},{"appFrontIcon":"cuIcon-news","buttons":["新增","查看","修改","删除"],"menu":"通知公告","tableName":"news"}],"menu":"系统管理"},{"child":[{"appFrontIcon":"cuIcon-phone","buttons":["查看","删除","导出","发货"],"menu":"已支付订单","tableName":"orders/已支付"},{"appFrontIcon":"cuIcon-goods","buttons":["查看","删除","导出"],"menu":"已完成订单","tableName":"orders/已完成"},{"appFrontIcon":"cuIcon-attentionfavor","buttons":["查看","删除","导出"],"menu":"已取消订单","tableName":"orders/已取消"},{"appFrontIcon":"cuIcon-send","buttons":["查看","删除","导出"],"menu":"已退款订单","tableName":"orders/已退款"},{"appFrontIcon":"cuIcon-explore","buttons":["查看","删除","导出"],"menu":"未支付订单","tableName":"orders/未支付"},{"appFrontIcon":"cuIcon-copy","buttons":["查看","删除","导出"],"menu":"已发货订单","tableName":"orders/已发货"}],"menu":"订单管理"}],"frontMenu":[{"child":[{"appFrontIcon":"cuIcon-attentionfavor","buttons":["查看"],"menu":"二手商品列表","menuJump":"列表","tableName":"ershoushangpin"}],"menu":"二手商品模块"},{"child":[{"appFrontIcon":"cuIcon-goods","buttons":["查看","领单"],"menu":"校园跑腿列表","menuJump":"列表","tableName":"xiaoyuanpaotui"}],"menu":"校园跑腿模块"}],"hasBackLogin":"是","hasBackRegister":"否","hasFrontLogin":"否","hasFrontRegister":"否","roleName":"管理员","tableName":"users"},{"backMenu":[{"child":[{"appFrontIcon":"cuIcon-flashlightopen","buttons":["新增","查看","删除"],"menu":"校园跑腿","menuJump":"列表","tableName":"xiaoyuanpaotui"}],"menu":"校园跑腿管理"},{"child":[{"appFrontIcon":"cuIcon-form","buttons":["查看","完成登记"],"menu":"跑腿领单","menuJump":"列表","tableName":"paotuilingdan"}],"menu":"跑腿领单管理"},{"child":[{"appFrontIcon":"cuIcon-time","buttons":["查看","支付","评价"],"menu":"完成订单","menuJump":"列表","tableName":"wanchengdingdan"}],"menu":"完成订单管理"},{"child":[{"appFrontIcon":"cuIcon-goodsnew","buttons":["查看"],"menu":"评价信息","menuJump":"列表","tableName":"pingjiaxinxi"}],"menu":"评价信息管理"}],"frontMenu":[{"child":[{"appFrontIcon":"cuIcon-attentionfavor","buttons":["查看"],"menu":"二手商品列表","menuJump":"列表","tableName":"ershoushangpin"}],"menu":"二手商品模块"},{"child":[{"appFrontIcon":"cuIcon-goods","buttons":["查看","领单"],"menu":"校园跑腿列表","menuJump":"列表","tableName":"xiaoyuanpaotui"}],"menu":"校园跑腿模块"}],"hasBackLogin":"否","hasBackRegister":"否","hasFrontLogin":"是","hasFrontRegister":"是","roleName":"用户","tableName":"yonghu"}],
			loginForm: {
				username: '',
				password: '',
				tableName: '',
				code: '',
			},
			role: '',
            roles: [],
			rules: {
				username: [
					{ required: true, message: '请输入账户', trigger: 'blur' }
				],
				password: [
					{ required: true, message: '请输入密码', trigger: 'blur' }
				]
			},
			codes: [{
				num: 1,
				color: '#000',
				rotate: '10deg',
				size: '16px'
			}, {
				num: 2,
				color: '#000',
				rotate: '10deg',
				size: '16px'
			}, {
				num: 3,
				color: '#000',
				rotate: '10deg',
				size: '16px'
			}, {
				num: 4,
				color: '#000',
				rotate: '10deg',
				size: '16px'
			}]
		}
	},
  components: {
  },
	created() {
        for(let item in this.roleMenus) {
            if(this.roleMenus[item].hasFrontLogin=='是') {
                this.roles.push(this.roleMenus[item]);
            }
        }
	},
	mounted() {
	},
    //方法集合
    methods: {
		randomString() {
			var len = 4;
			var chars = [
			  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
			  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
			  'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
			  'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
			  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2',
			  '3', '4', '5', '6', '7', '8', '9'
			]
			var colors = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
			var sizes = ['14', '15', '16', '17', '18']
			
			var output = []
			for (var i = 0; i < len; i++) {
			  // 随机验证码
			  var key = Math.floor(Math.random() * chars.length)
			  this.codes[i].num = chars[key]
			  // 随机验证码颜色
			  var code = '#'
			  for (var j = 0; j < 6; j++) {
			    var key = Math.floor(Math.random() * colors.length)
			    code += colors[key]
			  }
			  this.codes[i].color = code
			  // 随机验证码方向
			  var rotate = Math.floor(Math.random() * 45)
			  var plus = Math.floor(Math.random() * 2)
			  if (plus == 1) rotate = '-' + rotate
			  this.codes[i].rotate = 'rotate(' + rotate + 'deg)'
			  // 随机验证码字体大小
			  var size = Math.floor(Math.random() * sizes.length)
			  this.codes[i].size = sizes[size] + 'px'
			}
		},
      getCurrentRow(row) {
        this.role = row.roleName;
      },
      submitForm(formName) {
        if (this.roles.length!=1) {
            if (!this.role) {
                this.$message.error("请选择登录用户类型");
                return false;
            }
        } else {
            this.role = this.roles[0].roleName;
            this.loginForm.tableName = this.roles[0].tableName;
        }
        this.$refs[formName].validate((valid) => {
          if (valid) {
            this.$http.get(`${this.loginForm.tableName}/login`, {params: this.loginForm}).then(res => {
              if (res.data.code === 0) {
                localStorage.setItem('Token', res.data.token);
                localStorage.setItem('UserTableName', this.loginForm.tableName);
                localStorage.setItem('username', this.loginForm.username);
                localStorage.setItem('adminName', this.loginForm.username);
                localStorage.setItem('sessionTable', this.loginForm.tableName);
                localStorage.setItem('role', this.role);
                localStorage.setItem('keyPath', this.$config.indexNav.length+2);
                this.$router.push('/index/center');
                this.$message({
                  message: '登录成功',
                  type: 'success',
                  duration: 1500,
                });
              } else {
                this.$message.error(res.data.msg);
              }
            });
          } else {
            return false;
          }
        });
      },
      resetForm(formName) {
        this.$refs[formName].resetFields();
      }
    }
  }
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
	.container {
		position: relative;
		background: url(http://codegen.caihongy.cn/20210504/6f5e101948464749875bc28f19675784.jpg) no-repeat center top / 100% 100%;
		
		.el-form-item {
		  & /deep/ .el-form-item__content {
		    width: 100%;
		  }
		}
		
		.list-item /deep/ .el-input .el-input__inner {
			border: 1px solid #bdefb6;
			border-radius: 8px;
			padding: 0 10px;
			box-shadow: inset 0px 0px 12px 0px #c5f1c0;
			outline: 0px solid #efefef;
			color: #333;
			flex: 1;
			width: 100%;
			font-size: 14px;
			outline-offset: 4px;
			height: 40px;
		}
		
		.list-code /deep/ .el-input .el-input__inner {
			border: 1px solid #bdefb6;
			border-radius: 8px;
			padding: 0 10px;
			box-shadow: inset 0px 0px 12px 0px #c5f1c0;
			outline: none;
			color: #333;
			display: inline-block;
			vertical-align: middle;
			width: calc(100% - 110px);
			font-size: 14px;
			height: 40px;
		}
		
		.list-type /deep/ .el-radio__input .el-radio__inner {
			background: rgba(53, 53, 53, 0);
			border-color: #666;
		}
		.list-type /deep/ .el-radio__input.is-checked .el-radio__inner {
			background: #5bb450;
			border-color: #5bb450;
		}
		.list-type /deep/ .el-radio__label {
			color: #666;
			font-size: 14px;
		}
		.list-type /deep/ .el-radio__input.is-checked+.el-radio__label {
			color: #5bb450;
			font-size: 14px;
		}
	}
</style>
