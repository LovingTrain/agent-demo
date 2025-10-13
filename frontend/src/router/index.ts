// src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import Login from '@/components/Login.vue'
import Register from '@/components/Register.vue'
import Chatbot from '@/components/Chatbot.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: Login },
    { path: '/register', component: Register },
    {
      path: '/',
      component: Chatbot,
      beforeEnter: (to, from, next) => {
        const token = localStorage.getItem('token')
        if (!token) next('/login')
        else next()
      }
    },
    { path: '/:pathMatch(.*)*', redirect: '/' }
  ]
})

export default router