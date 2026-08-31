(()=>{
  const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
  const root=document.documentElement,course=document.body.dataset.course||'course',store='iitm-'+course+'-cheatsheet';
  const savedTheme=localStorage.getItem('study-hub-theme');if(savedTheme)root.dataset.theme=savedTheme;
  $('#theme').onclick=()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';localStorage.setItem('study-hub-theme',root.dataset.theme)};
  $('#print').onclick=()=>print();
  let week='all';
  function apply(){
    const q=$('#search').value.trim().toLowerCase();let visible=0;
    $$('.section[data-week]').forEach(s=>{const weekMatch=week==='all'||s.dataset.week.split(' ').includes(week),text=s.textContent.toLowerCase(),show=weekMatch&&(!q||text.includes(q));s.classList.toggle('hidden',!show);if(show)visible++});
    $('#match').textContent=visible+' section'+(visible===1?'':'s');
    $('#empty').style.display=visible?'none':'block';
  }
  $$('[data-filter]').forEach(b=>b.onclick=()=>{week=b.dataset.filter;$$('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));apply()});
  $('#search').oninput=apply;
  $$('.section[data-week]').forEach(s=>{const id=s.id,btn=$('.done',s),done=localStorage.getItem(store+'-'+id)==='1';if(done)btn.classList.add('checked');btn.textContent=done?'Reviewed':'Mark reviewed';btn.onclick=()=>{const next=!btn.classList.contains('checked');btn.classList.toggle('checked',next);btn.textContent=next?'Reviewed':'Mark reviewed';localStorage.setItem(store+'-'+id,next?'1':'0')}});
  apply();
})();
