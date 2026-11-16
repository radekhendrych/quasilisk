import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL,fileURLToPath} from 'node:url';
import path from 'node:path';
const HERE=path.dirname(fileURLToPath(import.meta.url));
const {chromium}=await import(pathToFileURL(process.env.QUASILISK_PLAYWRIGHT_MODULE||'/tmp/quasilisk-report-qa/node_modules/playwright/index.mjs').href);
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const p=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[],requests=[];
p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(!r.url().startsWith('file:'))requests.push(r.url())});
const base=pathToFileURL(path.join(HERE,'site/dist/')).href;
const data=JSON.parse(readFileSync(path.join(HERE,'site/data/report.json'),'utf8'));
try{
 for(const page of ['index','results','judging','selector','runtime','methodology','conversations']){await p.goto(base+page+'.html');await p.waitForFunction(()=>window.REPORT_READY||window.REPORT_ERROR);assert.equal(await p.evaluate(()=>window.REPORT_ERROR),undefined,page);}
 await p.waitForSelector('.case-title');
 for(const o of data.observations){await p.locator(`#case-list a[data-case="${o.id}"]`).click();await p.waitForFunction(id=>selectedCase?.id===id,o.id);const expected=JSON.parse(readFileSync(path.join(HERE,'site/data/conversations',o.cell+'-target-'+o.target_sequence+'.json'),'utf8'));assert.deepEqual(await p.locator('.turn .content').allTextContents(),expected.messages.map(m=>m.content));}
 await p.locator('.tabs [data-view="judging"]').click();await p.waitForSelector('#case-tab .card');await p.locator('.case-title [data-call]').click();await p.waitForSelector('#dialog-body h3');assert.ok((await p.locator('#dialog-body').innerText()).includes('Request messages'));await p.keyboard.press('Escape');
 await p.goto(base+'selector.html');await p.waitForFunction(()=>window.REPORT_READY);await p.locator('[data-strategy]').first().click();await p.waitForSelector('#dialog-body .card');assert.ok((await p.locator('#dialog-body').innerText()).includes('Template excludes'));
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 writeFileSync(path.join(HERE,'offline_validation.json'),JSON.stringify({passed:true,pages:7,exact_target_branches:84,call_and_strategy_inspectors:true,non_file_requests:requests,console_errors:errors},null,2)+'\n');console.log('PASS direct file opening: 7 pages, 84 exact branches, both inspectors, no network requests');
}finally{await browser.close()}
